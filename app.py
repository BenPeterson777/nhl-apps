"""
NHL Player Stats Dashboard - Shiny for Python
Data sourced from the official NHL Stats REST API (public, no key required).

steps to run app locally: 
1. open "Terminal" pane in bottom of VSCode -> from there, run the following lines:
    cd C:\Users\Ben\nhl_stats_app
    pip install -r requirements.txt
    shiny run app.py      
2. open local app link: http://127.0.0.1:8000/
"""

import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import requests
from shiny import App, reactive, render, ui

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
NHL_STATS_API = "https://api.nhle.com/stats/rest/en"
NHL_WEB_API   = "https://api-web.nhle.com/v1"

SEASONS = {
    "2025-26 (Current)": "20252026",
    "2024-25":           "20242025",
    "2023-24":           "20232024",
    "2022-23":           "20222023",
    "2021-22":           "20212022",
    "2020-21":           "20202021",
    "2019-20":           "20192020",
}

POSITION_LABELS = {
    "All Positions": "All",
    "Center (C)":    "C",
    "Left Wing (L)": "L",
    "Right Wing (R)":"R",
    "Defense (D)":   "D",
}

STAT_OPTIONS = {
    "Points":        "points",
    "Goals":         "goals",
    "Assists":       "assists",
    "Plus / Minus":  "plusMinus",
    "Shots":         "shots",
    "Penalty Mins":  "penaltyMinutes",
    "Shooting %":    "shootingPctg",
    "Points / Game": "pointsPerGame",
    "PP Goals":      "ppGoals",
    "PP Points":     "ppPoints",
    "SH Goals":      "shGoals",
    "EV Goals":      "evGoals",
    "Games Played":  "gamesPlayed",
}

DISPLAY_COLS = {
    "skaterFullName": "Player",
    "lastTeam":       "Team",
    "positionCode":   "Pos",
    "gamesPlayed":    "GP",
    "goals":          "G",
    "assists":        "A",
    "points":         "P",
    "plusMinus":      "+/-",
    "penaltyMinutes": "PIM",
    "shots":          "S",
    "shootingPctg":   "S%",
    "pointsPerGame":  "P/GP",
    "ppGoals":        "PPG",
    "ppPoints":       "PPP",
    "shGoals":        "SHG",
    "evGoals":        "EVG",
}

POS_COLORS = {"C": "#e74c3c", "L": "#3498db", "R": "#2ecc71", "D": "#f39c12"}
DARK_BG = "#1a1a2e"
CARD_BG = "#16213e"
ACCENT  = "#0f3460"
TEXT    = "#e0e0e0"


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────
def fetch_skater_stats(season_id: str) -> pd.DataFrame:
    """Pull all skater regular-season summary stats via the NHL Stats REST API."""
    url = f"{NHL_STATS_API}/skater/summary"
    all_rows: list[dict] = []
    start, limit = 0, 100

    while True:
        params = {
            "isAggregate": "false",
            "isGame":      "false",
            "sort": '[{"property":"points","direction":"DESC"},'
                    '{"property":"goals","direction":"DESC"}]',
            "start":       start,
            "limit":       limit,
            "cayenneExp":  f"seasonId={season_id} and gameTypeId=2",
        }
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            payload = r.json()
        except Exception as exc:
            print(f"[NHL API] fetch error at start={start}: {exc}")
            break

        rows  = payload.get("data", [])
        total = payload.get("total", 0)
        if not rows:
            break
        all_rows.extend(rows)
        start += limit
        if start >= total:
            break

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Normalise types
    for col in ("gamesPlayed", "goals", "assists", "points", "shots",
                "penaltyMinutes", "ppGoals", "ppPoints", "shGoals", "evGoals"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    if "shootingPctg" in df.columns:
        df["shootingPctg"] = (pd.to_numeric(df["shootingPctg"], errors="coerce") * 100).round(1)

    if "pointsPerGame" in df.columns:
        df["pointsPerGame"] = pd.to_numeric(df["pointsPerGame"], errors="coerce").round(3)

    # Derive the most recent (last) team for traded players, e.g. "MIN,VAN" → "VAN"
    if "teamAbbrevs" in df.columns:
        df["lastTeam"] = df["teamAbbrevs"].astype(str).str.split(",").str[-1].str.strip()

    return df


def apply_filters(df: pd.DataFrame, team: str, position: str, min_gp: int) -> pd.DataFrame:
    if df.empty:
        return df
    if "gamesPlayed" in df.columns:
        df = df[df["gamesPlayed"] >= min_gp]
    if team != "All" and "lastTeam" in df.columns:
        df = df[df["lastTeam"] == team]
    if position != "All" and "positionCode" in df.columns:
        df = df[df["positionCode"] == position]
    return df.reset_index(drop=True)


def make_display_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = {k: v for k, v in DISPLAY_COLS.items() if k in df.columns}
    out  = df[list(cols.keys())].rename(columns=cols).copy()
    out.insert(0, "Rank", range(1, len(out) + 1))
    return out


def style_fig(fig: plt.Figure, ax: plt.Axes) -> None:
    """Apply a consistent dark theme to a figure."""
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(ACCENT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.grid(axis="y", color=ACCENT, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
app_ui = ui.page_fluid(
    # Inline CSS for dark theme
    ui.tags.style("""
        body { background-color: #1a1a2e; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }
        .navbar, .navbar-default { background-color: #0f3460 !important; border: none; }
        .sidebar { background-color: #16213e; border-radius: 8px; padding: 15px; }
        .well { background-color: #16213e; border: 1px solid #0f3460; border-radius: 8px; }
        label { color: #b0b0c0 !important; font-size: 0.85rem; }
        select, input[type=range] { background-color: #0f3460; color: #e0e0e0; border: 1px solid #3498db; border-radius: 4px; }
        .nav-tabs { border-bottom: 2px solid #0f3460; }
        .nav-tabs > li > a { color: #b0b0c0; background-color: #16213e; border: 1px solid #0f3460; }
        .nav-tabs > li.active > a { background-color: #0f3460; color: #fff; }
        .tab-content { background-color: #1a1a2e; border: 1px solid #0f3460; border-top: none; border-radius: 0 0 8px 8px; padding: 15px; }
        h1, h4 { color: #3498db; }
        .stat-card { background-color: #16213e; border-left: 4px solid #3498db; border-radius: 6px; padding: 12px 16px; margin: 6px 0; }
        .stat-card .val { font-size: 1.6rem; font-weight: bold; color: #3498db; }
        .stat-card .lbl { font-size: 0.8rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
        .shiny-data-frame { font-size: 0.8rem; }
        .loading-note { color: #888; font-size: 0.85rem; font-style: italic; }
    """),

    ui.h1("NHL Player Stats Dashboard", style="padding: 15px 0 0 15px; margin: 0;"),
    ui.p("Live data · NHL Stats REST API", style="color:#888; font-size:0.8rem; padding: 0 0 10px 17px;"),

    ui.layout_sidebar(
        ui.sidebar(
            ui.h4("Filters", style="margin-top:0; color:#3498db;"),
            ui.input_select(
                "season", "Season",
                choices=list(SEASONS.keys()),
                selected="2024-25 (Current)",
            ),
            ui.input_select(
                "team", "Team",
                choices=["All"],
                selected="All",
            ),
            ui.input_select(
                "position", "Position",
                choices=list(POSITION_LABELS.keys()),
                selected="All Positions",
            ),
            ui.input_slider(
                "min_gp", "Min. Games Played",
                min=1, max=82, value=10, step=1,
            ),
            ui.hr(style="border-color: #0f3460;"),
            ui.h4("Chart Options", style="color:#3498db;"),
            ui.input_select(
                "chart_stat", "Highlight Stat",
                choices=list(STAT_OPTIONS.keys()),
                selected="Points",
            ),
            ui.input_slider(
                "top_n", "Top N Players (charts)",
                min=5, max=40, value=20, step=5,
            ),
            ui.hr(style="border-color: #0f3460;"),
            ui.output_ui("summary_cards"),
            class_="sidebar",
        ),

        ui.navset_tab(
            ui.nav_panel(
                "Data Table",
                ui.p("All filtered players sorted by points.", class_="loading-note"),
                ui.output_data_frame("stats_table"),
            ),
            ui.nav_panel(
                "Top Players",
                ui.output_plot("bar_chart", height="500px"),
            ),
            ui.nav_panel(
                "Goals vs Assists",
                ui.output_plot("scatter_plot", height="500px"),
            ),
            ui.nav_panel(
                "Points Distribution",
                ui.output_plot("dist_plot", height="500px"),
            ),
            ui.nav_panel(
                "Shooting %",
                ui.output_plot("shooting_plot", height="500px"),
            ),
        ),
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# Server
# ─────────────────────────────────────────────────────────────────────────────
def server(input, output, session):

    # ── Raw data (fetched once per season selection) ──────────────────────────
    @reactive.calc
    def raw_data() -> pd.DataFrame:
        season_id = SEASONS[input.season()]
        return fetch_skater_stats(season_id)

    # ── Dynamically populate the Team dropdown ────────────────────────────────
    @reactive.effect
    def _populate_teams():
        df = raw_data()
        if df.empty or "lastTeam" not in df.columns:
            choices = ["All"]
        else:
            teams   = sorted(df["lastTeam"].dropna().unique().tolist())
            choices = ["All"] + teams
        ui.update_select("team", choices=choices, selected="All")

    # ── Filtered dataset ──────────────────────────────────────────────────────
    @reactive.calc
    def filtered_data() -> pd.DataFrame:
        pos_code = POSITION_LABELS[input.position()]
        return apply_filters(raw_data(), input.team(), pos_code, input.min_gp())

    # ── Summary stat cards ────────────────────────────────────────────────────
    @render.ui
    def summary_cards():
        df = filtered_data()
        if df.empty:
            return ui.p("No data loaded.", style="color:#888;")

        n_players    = len(df)
        avg_pts      = round(df["points"].mean(), 1)  if "points"  in df.columns else "—"
        top_scorer   = df.iloc[0]["skaterFullName"]   if "skaterFullName" in df.columns else "—"
        top_pts      = df.iloc[0]["points"]           if "points"  in df.columns else "—"

        return ui.div(
            ui.div(
                ui.div(str(n_players), class_="val"),
                ui.div("Players", class_="lbl"),
                class_="stat-card",
            ),
            ui.div(
                ui.div(str(avg_pts), class_="val"),
                ui.div("Avg Points", class_="lbl"),
                class_="stat-card",
            ),
            ui.div(
                ui.div(f"{top_scorer}", class_="val", style="font-size:1rem;"),
                ui.div(f"Points Leader  ·  {top_pts} pts", class_="lbl"),
                class_="stat-card",
            ),
        )

    # ── Data table ────────────────────────────────────────────────────────────
    @render.data_frame
    def stats_table():
        df = filtered_data()
        if df.empty:
            return render.DataTable(pd.DataFrame({"Status": ["Loading data…"]}))
        return render.DataTable(
            make_display_df(df),
            filters=True,
            height="600px",
        )

    # ── Bar chart: Top N by selected stat ─────────────────────────────────────
    @render.plot
    def bar_chart():
        df   = filtered_data()
        stat = STAT_OPTIONS[input.chart_stat()]
        n    = input.top_n()

        fig, ax = plt.subplots(figsize=(10, 6))
        style_fig(fig, ax)

        if df.empty or stat not in df.columns:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                    color=TEXT, transform=ax.transAxes, fontsize=14)
            return fig

        top = df.nlargest(n, stat)[["skaterFullName", "lastTeam", "positionCode", stat]].copy()
        top = top.iloc[::-1]  # flip so highest is at top

        colors = [POS_COLORS.get(p, "#95a5a6") for p in top["positionCode"]]
        bars   = ax.barh(top["skaterFullName"], top[stat], color=colors, edgecolor="none", height=0.7)

        # Value labels
        for bar, val in zip(bars, top[stat]):
            ax.text(bar.get_width() + max(top[stat]) * 0.01, bar.get_y() + bar.get_height() / 2,
                    str(val), va="center", ha="left", color=TEXT, fontsize=8)

        ax.set_xlabel(input.chart_stat(), fontsize=10)
        ax.set_title(f"Top {n} Players  ·  {input.chart_stat()}  ·  {input.season()}", fontsize=12, pad=12)
        ax.tick_params(axis="y", labelsize=8)

        # Legend
        patches = [mpatches.Patch(color=c, label=p) for p, c in POS_COLORS.items()]
        ax.legend(handles=patches, loc="lower right", framealpha=0.3,
                  facecolor=CARD_BG, edgecolor=ACCENT, labelcolor=TEXT, fontsize=8)

        fig.tight_layout()
        return fig

    # ── Scatter: Goals vs Assists ─────────────────────────────────────────────
    @render.plot
    def scatter_plot():
        df = filtered_data()
        fig, ax = plt.subplots(figsize=(10, 6))
        style_fig(fig, ax)

        if df.empty or "goals" not in df.columns or "assists" not in df.columns:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                    color=TEXT, transform=ax.transAxes, fontsize=14)
            return fig

        for pos, grp in df.groupby("positionCode"):
            ax.scatter(
                grp["assists"], grp["goals"],
                c=POS_COLORS.get(pos, "#95a5a6"),
                label=pos, alpha=0.75, s=50, edgecolors="none",
            )

        # Label top 10 scorers
        top10 = df.nlargest(10, "points")
        for _, row in top10.iterrows():
            ax.annotate(
                row["skaterFullName"].split()[-1],
                xy=(row["assists"], row["goals"]),
                xytext=(4, 2), textcoords="offset points",
                color=TEXT, fontsize=7, alpha=0.9,
            )

        # 45-degree "pace" line (equal G & A)
        lim = max(df["assists"].max(), df["goals"].max()) * 1.05
        ax.plot([0, lim], [0, lim], "--", color="#888", linewidth=0.8, alpha=0.5, label="G = A pace")

        ax.set_xlabel("Assists", fontsize=10)
        ax.set_ylabel("Goals", fontsize=10)
        ax.set_title(f"Goals vs Assists  ·  {input.season()}  ·  min {input.min_gp()} GP", fontsize=12, pad=12)
        ax.legend(framealpha=0.3, facecolor=CARD_BG, edgecolor=ACCENT, labelcolor=TEXT, fontsize=9)

        fig.tight_layout()
        return fig

    # ── Distribution: Points histogram by position ────────────────────────────
    @render.plot
    def dist_plot():
        df = filtered_data()
        fig, ax = plt.subplots(figsize=(10, 6))
        style_fig(fig, ax)

        if df.empty or "points" not in df.columns:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                    color=TEXT, transform=ax.transAxes, fontsize=14)
            return fig

        bins = np.linspace(0, df["points"].max() + 5, 30)

        for pos in ["C", "L", "R", "D"]:
            sub = df[df["positionCode"] == pos]["points"]
            if sub.empty:
                continue
            ax.hist(sub, bins=bins, alpha=0.55, color=POS_COLORS[pos],
                    label=f"{pos}  (n={len(sub)}, avg={sub.mean():.1f})", edgecolor="none")

        ax.set_xlabel("Points", fontsize=10)
        ax.set_ylabel("# of Players", fontsize=10)
        ax.set_title(f"Points Distribution by Position  ·  {input.season()}", fontsize=12, pad=12)
        ax.legend(framealpha=0.3, facecolor=CARD_BG, edgecolor=ACCENT, labelcolor=TEXT, fontsize=9)

        fig.tight_layout()
        return fig

    # ── Shooting %: scatter of shots vs goals, sized by GP ───────────────────
    @render.plot
    def shooting_plot():
        df = filtered_data()
        fig, ax = plt.subplots(figsize=(10, 6))
        style_fig(fig, ax)

        if df.empty or "shots" not in df.columns or "shootingPctg" not in df.columns:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                    color=TEXT, transform=ax.transAxes, fontsize=14)
            return fig

        # At least 20 shots for a meaningful shooting %
        sub = df[df["shots"] >= 20].copy()
        if sub.empty:
            ax.text(0.5, 0.5, "Not enough data (need ≥ 20 shots)", ha="center", va="center",
                    color=TEXT, transform=ax.transAxes, fontsize=14)
            return fig

        colors = [POS_COLORS.get(p, "#95a5a6") for p in sub["positionCode"]]
        sizes  = np.clip(sub["gamesPlayed"] * 1.2, 20, 120)

        sc = ax.scatter(sub["shots"], sub["shootingPctg"],
                        c=colors, s=sizes, alpha=0.7, edgecolors="none")

        # League-average line
        lg_avg = (sub["goals"].sum() / sub["shots"].sum() * 100) if sub["shots"].sum() > 0 else 0
        ax.axhline(lg_avg, color="#e0e0e0", linewidth=0.9, linestyle="--",
                   label=f"Filtered avg  {lg_avg:.1f}%", alpha=0.7)

        # Label top 8 by goals
        for _, row in sub.nlargest(8, "goals").iterrows():
            ax.annotate(row["skaterFullName"].split()[-1],
                        xy=(row["shots"], row["shootingPctg"]),
                        xytext=(4, 2), textcoords="offset points",
                        color=TEXT, fontsize=7)

        ax.set_xlabel("Shots on Goal", fontsize=10)
        ax.set_ylabel("Shooting %", fontsize=10)
        ax.set_title(f"Shots vs Shooting %  ·  {input.season()}  ·  (bubble = GP)", fontsize=12, pad=12)

        patches = [mpatches.Patch(color=c, label=p) for p, c in POS_COLORS.items()]
        patches.append(mpatches.Patch(color="none", label=f"Filtered avg {lg_avg:.1f}%",
                                       linestyle="--", edgecolor="white"))
        ax.legend(handles=patches, framealpha=0.3, facecolor=CARD_BG,
                  edgecolor=ACCENT, labelcolor=TEXT, fontsize=9)

        fig.tight_layout()
        return fig


app = App(app_ui, server)
