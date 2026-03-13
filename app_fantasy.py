"""
NHL Fantasy Hockey Dashboard  –  Shiny for Python
Ranks players by your league's custom fantasy scoring system.

Fantasy Scoring Rules (from screenshot):
  SKATERS
    Goals          +3 pts each
    Hat Trick      +5 pts bonus  (excluded – not in aggregate API data)
    PP Point       +1 pt each
    SH Point       +1 pt each
    Assist         +2 pts each
    Plus/Minus     +0.5 pts per +/-

  GOALIES
    Win            +5 pts each
    Shutout        +6 pts each
    Saves          +0.0667 pts each  (1 pt per 15 saves)
    Goals Against  -0.5 pts each

steps to run locally:
  cd C:\\Users\\Ben\\nhl_stats_app
  shiny run app_fantasy.py
  open http://127.0.0.1:8000/
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import requests
from shiny import App, reactive, render, ui

# ─────────────────────────────────────────────────────────────────────────────
# Fantasy scoring constants
# ─────────────────────────────────────────────────────────────────────────────
GOAL_PTS      = 3.0
PP_POINT_PTS  = 1.0
SH_POINT_PTS  = 1.0
ASSIST_PTS    = 2.0
PM_PTS        = 0.5    # per unit of plus/minus

WIN_PTS       = 5.0
SHUTOUT_PTS   = 6.0
SAVE_PTS      = 1 / 15  # ≈ 0.0667 per save
GA_PTS        = -0.5    # per goal against

NHL_STATS_API = "https://api.nhle.com/stats/rest/en"
NHL_WEB_API   = "https://api-web.nhle.com/v1"
REGULAR_SEASON_GP = 82

# ─────────────────────────────────────────────────────────────────────────────
# My Roster  (team: "The Dirty Bubble")
# Update these lists whenever you make trades / pickups.
# ─────────────────────────────────────────────────────────────────────────────
MY_SKATERS = {
    # Forwards
    "Wyatt Johnston",
    "Brandon Hagel",
    "Seth Jarvis",
    "Filip Forsberg",
    "Nikolaj Ehlers",
    "Steven Stamkos",
    "Leo Carlsson",
    "Jack Hughes",
    "Jesper Bratt",
    "Brayden Point",
    "Bryan Rust",
    "Patrick Kane",
    "Roope Hintz",
    "Mark Stone",
    "Dylan Holloway",
    # Defense
    "Cale Makar",
    "Darren Raddysh",
    "Rasmus Dahlin",
    "Shea Theodore",
}

MY_GOALIES = {
    "Karel Vejmelka",
    "Lukas Dostal",
    "Logan Thompson",
    "Igor Shesterkin",
}

SEASONS = {
    "2025-26 (Current)": "20252026",
    "2024-25":           "20242025",
    "2023-24":           "20232024",
    "2022-23":           "20222023",
    "2021-22":           "20212022",
    "2020-21":           "20202021",
}

DARK_BG  = "#1a1a2e"
CARD_BG  = "#16213e"
ACCENT   = "#0f3460"
TEXT     = "#e0e0e0"

SKATER_COLORS = {
    "Goals":   "#e74c3c",
    "Assists": "#3498db",
    "PP Pts":  "#9b59b6",
    "SH Pts":  "#1abc9c",
    "+/-":     "#f39c12",
}
GOALIE_COLORS = {
    "Wins":     "#3498db",
    "Shutouts": "#9b59b6",
    "Saves":    "#2ecc71",
    "GA":       "#e74c3c",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data fetching
# ─────────────────────────────────────────────────────────────────────────────
def fetch_team_games_remaining() -> dict:
    """Return {team_abbrev: games_remaining} using live standings.
    Far more accurate than 82 - player_gp because it accounts for injuries
    and the fact that teams haven't all played the same number of games."""
    try:
        r = requests.get(f"{NHL_WEB_API}/standings/now", timeout=15)
        r.raise_for_status()
        result = {}
        for s in r.json().get("standings", []):
            abbrev = s.get("teamAbbrev", {}).get("default", "")
            team_gp = s.get("gamesPlayed", 0)
            if abbrev:
                result[abbrev] = max(0, REGULAR_SEASON_GP - team_gp)
        return result
    except Exception as exc:
        print(f"[NHL API] standings error: {exc}")
        return {}


def _paginate(endpoint: str, season_id: str, extra_exp: str = "") -> pd.DataFrame:
    url = f"{NHL_STATS_API}/{endpoint}"
    exp = f"seasonId={season_id} and gameTypeId=2"
    if extra_exp:
        exp += f" and {extra_exp}"
    all_rows, start, limit = [], 0, 100
    while True:
        params = {
            "isAggregate": "false",
            "isGame":      "false",
            "sort":        '[{"property":"points","direction":"DESC"}]',
            "start":       start,
            "limit":       limit,
            "cayenneExp":  exp,
        }
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            payload = r.json()
        except Exception as exc:
            print(f"[NHL API] {endpoint} error at start={start}: {exc}")
            break
        rows = payload.get("data", [])
        if not rows:
            break
        all_rows.extend(rows)
        start += limit
        if start >= payload.get("total", 0):
            break
    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


def fetch_skaters(season_id: str, team_rem: dict) -> pd.DataFrame:
    df = _paginate("skater/summary", season_id)
    if df.empty:
        return df

    int_cols = ["gamesPlayed", "goals", "assists", "points", "shots",
                "penaltyMinutes", "ppGoals", "ppPoints", "shGoals", "shPoints",
                "evGoals", "plusMinus", "gameWinningGoals", "otGoals"]
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    if "shootingPctg" in df.columns:
        df["shootingPctg"] = (pd.to_numeric(df["shootingPctg"], errors="coerce") * 100).round(1)

    # Derive last/current team
    if "teamAbbrevs" in df.columns:
        df["lastTeam"] = df["teamAbbrevs"].astype(str).str.split(",").str[-1].str.strip()

    # Position group
    if "positionCode" in df.columns:
        df["posGroup"] = df["positionCode"].map(
            lambda p: "D" if p == "D" else "F"
        )

    # ── Fantasy score components ──────────────────────────────────────────
    df["fpts_goals"]   = df.get("goals",    pd.Series(0, index=df.index)) * GOAL_PTS
    df["fpts_assists"] = df.get("assists",  pd.Series(0, index=df.index)) * ASSIST_PTS
    df["fpts_pp"]      = df.get("ppPoints", pd.Series(0, index=df.index)) * PP_POINT_PTS
    df["fpts_sh"]      = df.get("shPoints", pd.Series(0, index=df.index)) * SH_POINT_PTS
    df["fpts_pm"]      = df.get("plusMinus",pd.Series(0, index=df.index)) * PM_PTS

    df["fantasy_pts"] = (
        df["fpts_goals"] + df["fpts_assists"] +
        df["fpts_pp"]    + df["fpts_sh"]      + df["fpts_pm"]
    ).round(1)

    # ── Per-game & projection (uses actual team schedule remaining) ───────
    gp = df["gamesPlayed"].replace(0, np.nan)
    df["fpts_per_game"] = (df["fantasy_pts"] / gp).round(2)
    if team_rem and "lastTeam" in df.columns:
        df["games_remaining"] = df["lastTeam"].map(team_rem).fillna(0).clip(lower=0).astype(int)
    else:
        df["games_remaining"] = (REGULAR_SEASON_GP - df["gamesPlayed"]).clip(lower=0)
    df["proj_season_pts"] = (
        df["fantasy_pts"] + df["fpts_per_game"] * df["games_remaining"]
    ).round(1)

    return df.sort_values("fantasy_pts", ascending=False).reset_index(drop=True)


def fetch_goalies(season_id: str, team_rem: dict) -> pd.DataFrame:
    url  = f"{NHL_STATS_API}/goalie/summary"
    all_rows, start, limit = [], 0, 100
    while True:
        params = {
            "isAggregate": "false",
            "isGame":      "false",
            "sort":        '[{"property":"wins","direction":"DESC"}]',
            "start":       start,
            "limit":       limit,
            "cayenneExp":  f"seasonId={season_id} and gameTypeId=2",
        }
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            payload = r.json()
        except Exception as exc:
            print(f"[NHL API] goalie error at start={start}: {exc}")
            break
        rows = payload.get("data", [])
        if not rows:
            break
        all_rows.extend(rows)
        start += limit
        if start >= payload.get("total", 0):
            break

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    int_cols = ["gamesPlayed", "gamesStarted", "wins", "losses",
                "otLosses", "shutouts", "saves", "goalsAgainst"]
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    if "teamAbbrevs" in df.columns:
        df["lastTeam"] = df["teamAbbrevs"].astype(str).str.split(",").str[-1].str.strip()

    # ── Fantasy score components ──────────────────────────────────────────
    df["fpts_wins"]     = df.get("wins",         pd.Series(0, index=df.index)) * WIN_PTS
    df["fpts_shutouts"] = df.get("shutouts",      pd.Series(0, index=df.index)) * SHUTOUT_PTS
    df["fpts_saves"]    = (df.get("saves",        pd.Series(0, index=df.index)) * SAVE_PTS).round(1)
    df["fpts_ga"]       = df.get("goalsAgainst",  pd.Series(0, index=df.index)) * GA_PTS

    df["fantasy_pts"] = (
        df["fpts_wins"] + df["fpts_shutouts"] + df["fpts_saves"] + df["fpts_ga"]
    ).round(1)

    gp = df["gamesPlayed"].replace(0, np.nan)
    df["fpts_per_game"] = (df["fantasy_pts"] / gp).round(2)
    if team_rem and "lastTeam" in df.columns:
        df["games_remaining"] = df["lastTeam"].map(team_rem).fillna(0).clip(lower=0).astype(int)
    else:
        df["games_remaining"] = (REGULAR_SEASON_GP - df["gamesPlayed"]).clip(lower=0)
    df["proj_season_pts"] = (
        df["fantasy_pts"] + df["fpts_per_game"] * df["games_remaining"]
    ).round(1)

    return df.sort_values("fantasy_pts", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def style_fig(fig, ax):
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(ACCENT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.grid(axis="x", color=ACCENT, linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)


def make_skater_display(df: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "skaterFullName": "Player",
        "lastTeam":       "Team",
        "posGroup":       "Pos",
        "gamesPlayed":    "GP",
        "goals":          "G",
        "assists":        "A",
        "ppPoints":       "PPP",
        "shPoints":       "SHP",
        "plusMinus":      "+/-",
        "fantasy_pts":    "Fantasy Pts",
        "fpts_per_game":  "Fpts/GP",
        "games_remaining":"GP Rem.",
        "proj_season_pts":"Proj. Total",
    }
    available = {k: v for k, v in cols.items() if k in df.columns}
    out = df[list(available.keys())].rename(columns=available).copy()
    out.insert(0, "Rank", range(1, len(out) + 1))
    return out


def make_goalie_display(df: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "goaltenderFullName": "Goalie",
        "lastTeam":           "Team",
        "gamesPlayed":        "GP",
        "wins":               "W",
        "shutouts":           "SO",
        "saves":              "SV",
        "goalsAgainst":       "GA",
        "fantasy_pts":        "Fantasy Pts",
        "fpts_per_game":      "Fpts/GP",
        "games_remaining":    "GP Rem.",
        "proj_season_pts":    "Proj. Total",
    }
    available = {k: v for k, v in cols.items() if k in df.columns}
    out = df[list(available.keys())].rename(columns=available).copy()
    out.insert(0, "Rank", range(1, len(out) + 1))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
app_ui = ui.page_fluid(
    ui.tags.style("""
        body { background-color: #1a1a2e; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }
        .sidebar { background-color: #16213e; border-radius: 8px; padding: 15px; }
        .well { background-color: #16213e; border: 1px solid #0f3460; border-radius: 8px; }
        label { color: #b0b0c0 !important; font-size: 0.85rem; }
        select, input[type=range] { background-color: #0f3460 !important; color: #e0e0e0 !important;
            border: 1px solid #3498db !important; border-radius: 4px; }
        .nav-tabs { border-bottom: 2px solid #0f3460; }
        .nav-tabs > li > a { color: #b0b0c0; background-color: #16213e; border: 1px solid #0f3460; }
        .nav-tabs > li.active > a { background-color: #0f3460; color: #fff; }
        .tab-content { background-color: #1a1a2e; border: 1px solid #0f3460;
            border-top: none; border-radius: 0 0 8px 8px; padding: 15px; }
        h1, h2, h4 { color: #3498db; }
        .stat-card { background-color: #16213e; border-left: 4px solid #f39c12;
            border-radius: 6px; padding: 10px 14px; margin: 5px 0; }
        .stat-card .val { font-size: 1.4rem; font-weight: bold; color: #f39c12; }
        .stat-card .lbl { font-size: 0.75rem; color: #888;
            text-transform: uppercase; letter-spacing: 1px; }
        .note { color: #e67e22; font-size: 0.8rem; font-style: italic; padding: 4px 0; }
        .shiny-data-frame { font-size: 0.8rem; }
        hr { border-color: #0f3460; }
        input[type=checkbox] { accent-color: #f39c12; width: 15px; height: 15px; cursor: pointer; }
        .checkbox label { color: #f39c12 !important; font-weight: bold; font-size: 0.9rem; }
    """),

    ui.h1("NHL Fantasy Hockey Rankings", style="padding:15px 0 0 15px; margin:0;"),
    ui.p(
        "Fantasy pts scored using your league's custom rules  ·  Projected totals based on current pace × remaining games",
        style="color:#888; font-size:0.8rem; padding:0 0 10px 17px;"
    ),
    ui.p(
        "Note: Hat Trick bonuses (+5) are excluded — not available in aggregate stats.",
        style="color:#e67e22; font-size:0.8rem; padding:0 0 6px 17px; font-style:italic;"
    ),

    ui.layout_sidebar(
        ui.sidebar(
            ui.h4("Filters", style="margin-top:0;"),
            ui.input_select(
                "season", "Season",
                choices=list(SEASONS.keys()),
                selected="2025-26 (Current)",
            ),
            ui.input_select(
                "player_type", "Player Type",
                choices=["Skaters — All", "Skaters — Forwards", "Skaters — Defense", "Goalies"],
                selected="Skaters — All",
            ),
            ui.input_checkbox(
                "my_roster_only", "My Roster Only (The Dirty Bubble)",
                value=False,
            ),
            ui.input_select("team", "Team", choices=["All"], selected="All"),
            ui.input_slider("min_gp", "Min. Games Played", min=1, max=82, value=10, step=1),
            ui.hr(),
            ui.h4("Chart Options", style="color:#3498db;"),
            ui.input_slider("top_n", "Top N Players", min=5, max=30, value=15, step=5),
            ui.hr(),
            ui.output_ui("summary_cards"),
            class_="sidebar",
        ),

        ui.navset_tab(
            ui.nav_panel(
                "Fantasy Rankings",
                ui.p("Ranked by total fantasy points accumulated this season.", class_="note",
                     style="color:#888; font-style:italic;"),
                ui.output_data_frame("rankings_table"),
            ),
            ui.nav_panel(
                "Top Fantasy Players",
                ui.output_plot("bar_total", height="520px"),
            ),
            ui.nav_panel(
                "Projected Season Total",
                ui.output_plot("bar_proj", height="520px"),
            ),
            ui.nav_panel(
                "Points Breakdown",
                ui.output_plot("bar_stacked", height="520px"),
            ),
            ui.nav_panel(
                "Rate vs Volume",
                ui.output_plot("scatter_rate", height="520px"),
            ),
        ),
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# Server
# ─────────────────────────────────────────────────────────────────────────────
def server(input, output, session):

    @reactive.calc
    def is_goalie() -> bool:
        return input.player_type() == "Goalies"

    # ── Raw data ──────────────────────────────────────────────────────────────
    @reactive.calc
    def team_remaining() -> dict:
        return fetch_team_games_remaining()

    @reactive.calc
    def raw_skaters() -> pd.DataFrame:
        return fetch_skaters(SEASONS[input.season()], team_remaining())

    @reactive.calc
    def raw_goalies() -> pd.DataFrame:
        return fetch_goalies(SEASONS[input.season()], team_remaining())

    # ── Team dropdown ─────────────────────────────────────────────────────────
    @reactive.effect
    def _update_teams():
        df = raw_goalies() if is_goalie() else raw_skaters()
        if df.empty or "lastTeam" not in df.columns:
            choices = ["All"]
        else:
            choices = ["All"] + sorted(df["lastTeam"].dropna().unique().tolist())
        ui.update_select("team", choices=choices, selected="All")

    # ── Filtered data ─────────────────────────────────────────────────────────
    @reactive.calc
    def filtered() -> pd.DataFrame:
        if is_goalie():
            df = raw_goalies()
            name_col = "goaltenderFullName"
            roster   = MY_GOALIES
        else:
            df = raw_skaters()
            name_col = "skaterFullName"
            roster   = MY_SKATERS
            pt = input.player_type()
            if pt == "Skaters — Forwards" and "posGroup" in df.columns:
                df = df[df["posGroup"] == "F"]
            elif pt == "Skaters — Defense" and "posGroup" in df.columns:
                df = df[df["posGroup"] == "D"]

        if df.empty:
            return df

        # Roster filter (applied before GP/team filters so IR players still show)
        if input.my_roster_only() and name_col in df.columns:
            roster_lower = {n.lower() for n in roster}
            df = df[df[name_col].str.lower().isin(roster_lower)]

        df = df[df["gamesPlayed"] >= input.min_gp()]
        if input.team() != "All" and "lastTeam" in df.columns:
            df = df[df["lastTeam"] == input.team()]
        return df.reset_index(drop=True)

    # ── Summary cards ─────────────────────────────────────────────────────────
    @render.ui
    def summary_cards():
        df = filtered()
        if df.empty:
            return ui.p("No data loaded.", style="color:#888;")

        n     = len(df)
        avg   = round(df["fantasy_pts"].mean(), 1)
        name_col = "goaltenderFullName" if is_goalie() else "skaterFullName"
        leader = df.iloc[0][name_col] if name_col in df.columns else "—"
        top_pts = df.iloc[0]["fantasy_pts"]
        proj_leader = df.sort_values("proj_season_pts", ascending=False).iloc[0]
        proj_name   = proj_leader[name_col] if name_col in df.columns else "—"
        proj_pts    = proj_leader["proj_season_pts"]

        return ui.div(
            ui.div(ui.div(str(n), class_="val"), ui.div("Players", class_="lbl"), class_="stat-card"),
            ui.div(ui.div(str(avg), class_="val"), ui.div("Avg Fantasy Pts", class_="lbl"), class_="stat-card"),
            ui.div(
                ui.div(leader, class_="val", style="font-size:0.95rem;"),
                ui.div(f"Points Leader · {top_pts} fpts", class_="lbl"),
                class_="stat-card",
            ),
            ui.div(
                ui.div(proj_name, class_="val", style="font-size:0.95rem;"),
                ui.div(f"Projected Leader · {proj_pts} fpts", class_="lbl"),
                class_="stat-card",
            ),
        )

    # ── Rankings table ────────────────────────────────────────────────────────
    @render.data_frame
    def rankings_table():
        df = filtered()
        if df.empty:
            return render.DataTable(pd.DataFrame({"Status": ["Loading…"]}))
        display = make_goalie_display(df) if is_goalie() else make_skater_display(df)
        return render.DataTable(display, filters=True, height="580px")

    # ── Bar: current total fantasy pts ────────────────────────────────────────
    @render.plot
    def bar_total():
        df = filtered()
        n  = input.top_n()
        fig, ax = plt.subplots(figsize=(10, 6))
        style_fig(fig, ax)

        if df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", color=TEXT, transform=ax.transAxes)
            return fig

        name_col = "goaltenderFullName" if is_goalie() else "skaterFullName"
        top = df.nlargest(n, "fantasy_pts").iloc[::-1]

        color = "#3498db" if is_goalie() else "#e74c3c"
        bars  = ax.barh(top[name_col], top["fantasy_pts"], color=color, edgecolor="none", height=0.7)

        max_val = top["fantasy_pts"].max()
        for bar, val in zip(bars, top["fantasy_pts"]):
            ax.text(bar.get_width() + max_val * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}", va="center", color=TEXT, fontsize=8)

        ax.set_xlabel("Fantasy Points", fontsize=10)
        ax.set_title(f"Top {n} Fantasy Points  ·  {input.season()}", fontsize=12, pad=12)
        ax.tick_params(axis="y", labelsize=8)
        fig.tight_layout()
        return fig

    # ── Bar: projected season total ───────────────────────────────────────────
    @render.plot
    def bar_proj():
        df = filtered()
        n  = input.top_n()
        fig, ax = plt.subplots(figsize=(10, 6))
        style_fig(fig, ax)

        if df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", color=TEXT, transform=ax.transAxes)
            return fig

        name_col = "goaltenderFullName" if is_goalie() else "skaterFullName"
        top = df.nlargest(n, "proj_season_pts").iloc[::-1]

        # Split bar: earned vs projected remaining
        earned = top["fantasy_pts"]
        proj   = (top["proj_season_pts"] - top["fantasy_pts"]).clip(lower=0)

        ax.barh(top[name_col], earned, color="#2ecc71", edgecolor="none", height=0.7, label="Earned so far")
        ax.barh(top[name_col], proj,   left=earned, color="#27ae60", alpha=0.45,
                edgecolor="none", height=0.7, label="Projected remaining")

        max_val = top["proj_season_pts"].max()
        for i, (_, row) in enumerate(top.iterrows()):
            ax.text(row["proj_season_pts"] + max_val * 0.01, i,
                    f"{row['proj_season_pts']:.0f}", va="center", color=TEXT, fontsize=8)

        ax.set_xlabel("Fantasy Points", fontsize=10)
        ax.set_title(f"Projected Season Total (pace × remaining games)  ·  {input.season()}", fontsize=12, pad=12)
        ax.tick_params(axis="y", labelsize=8)
        ax.legend(framealpha=0.3, facecolor=CARD_BG, edgecolor=ACCENT, labelcolor=TEXT, fontsize=9)
        fig.tight_layout()
        return fig

    # ── Stacked bar: breakdown of what's driving fantasy score ───────────────
    @render.plot
    def bar_stacked():
        df = filtered()
        n  = input.top_n()
        fig, ax = plt.subplots(figsize=(10, 6))
        style_fig(fig, ax)

        if df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", color=TEXT, transform=ax.transAxes)
            return fig

        name_col = "goaltenderFullName" if is_goalie() else "skaterFullName"
        top = df.nlargest(n, "fantasy_pts").iloc[::-1]
        names = top[name_col].tolist()

        if is_goalie():
            components = [
                ("Wins",     top["fpts_wins"],     GOALIE_COLORS["Wins"]),
                ("Shutouts", top["fpts_shutouts"],  GOALIE_COLORS["Shutouts"]),
                ("Saves",    top["fpts_saves"],     GOALIE_COLORS["Saves"]),
                ("GA",       top["fpts_ga"],        GOALIE_COLORS["GA"]),
            ]
        else:
            components = [
                ("Goals",   top["fpts_goals"],   SKATER_COLORS["Goals"]),
                ("Assists", top["fpts_assists"], SKATER_COLORS["Assists"]),
                ("PP Pts",  top["fpts_pp"],      SKATER_COLORS["PP Pts"]),
                ("SH Pts",  top["fpts_sh"],      SKATER_COLORS["SH Pts"]),
                ("+/-",     top["fpts_pm"],      SKATER_COLORS["+/-"]),
            ]

        left = np.zeros(len(top))
        for label, vals, color in components:
            vals_arr = vals.values
            # For negative values (GA), handle separately
            pos_vals = np.where(vals_arr >= 0, vals_arr, 0)
            neg_vals = np.where(vals_arr < 0,  vals_arr, 0)
            ax.barh(names, pos_vals, left=left,       color=color, edgecolor="none", height=0.7, label=label)
            ax.barh(names, neg_vals, left=left+pos_vals, color=color, edgecolor="none", height=0.7, alpha=0.5)
            left = left + pos_vals

        ax.set_xlabel("Fantasy Points", fontsize=10)
        ax.set_title(f"Fantasy Points Breakdown (what's driving the score)  ·  {input.season()}", fontsize=12, pad=12)
        ax.tick_params(axis="y", labelsize=8)
        ax.legend(framealpha=0.3, facecolor=CARD_BG, edgecolor=ACCENT, labelcolor=TEXT, fontsize=9,
                  loc="lower right")
        fig.tight_layout()
        return fig

    # ── Scatter: fpts/game vs total fpts (rate vs volume) ────────────────────
    @render.plot
    def scatter_rate():
        df = filtered()
        fig, ax = plt.subplots(figsize=(10, 6))
        style_fig(fig, ax)

        if df.empty or "fpts_per_game" not in df.columns:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", color=TEXT, transform=ax.transAxes)
            return fig

        name_col = "goaltenderFullName" if is_goalie() else "skaterFullName"
        color    = "#3498db" if is_goalie() else "#e74c3c"

        sizes = np.clip(df["gamesPlayed"] * 1.5, 15, 120)
        ax.scatter(df["fantasy_pts"], df["fpts_per_game"],
                   c=color, s=sizes, alpha=0.65, edgecolors="none")

        # Label top 12 by total pts
        for _, row in df.nlargest(12, "fantasy_pts").iterrows():
            ax.annotate(row[name_col].split()[-1],
                        xy=(row["fantasy_pts"], row["fpts_per_game"]),
                        xytext=(4, 2), textcoords="offset points",
                        color=TEXT, fontsize=7, alpha=0.9)

        # Quadrant lines at medians
        med_x = df["fantasy_pts"].median()
        med_y = df["fpts_per_game"].median()
        ax.axvline(med_x, color="#888", linewidth=0.8, linestyle="--", alpha=0.6, label=f"Median pts ({med_x:.0f})")
        ax.axhline(med_y, color="#f39c12", linewidth=0.8, linestyle="--", alpha=0.6, label=f"Median rate ({med_y:.2f}/gp)")

        ax.set_xlabel("Total Fantasy Points (volume)", fontsize=10)
        ax.set_ylabel("Fantasy Points per Game (rate)", fontsize=10)
        ax.set_title(
            f"Rate vs Volume  ·  {input.season()}  ·  bubble size = GP\n"
            "Top-right = elite  ·  Top-left = hot streak/fewer games  ·  Bottom-right = reliable but slow",
            fontsize=10, pad=10
        )
        ax.legend(framealpha=0.3, facecolor=CARD_BG, edgecolor=ACCENT, labelcolor=TEXT, fontsize=9)
        fig.tight_layout()
        return fig


app = App(app_ui, server)
