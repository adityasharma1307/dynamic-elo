import os
import numpy as np
import pandas as pd

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
CHARTING_REPO  = os.path.join(BASE_DIR, "data", "tennis_MatchChartingProject")
ATP_REPO       = os.path.join(BASE_DIR, "data", "tennis_atp")
EXISTING_GS    = os.path.join(BASE_DIR, "data", "cleaned", "atp_grand_slams_clean.parquet")
OUTPUT_PARQUET = os.path.join(BASE_DIR, "data", "cleaned", "atp_grand_slams_1990_2025.parquet")

SLAM_SURF  = {
    "Australian Open": "Hard",
    "Roland Garros":   "Clay",
    "Wimbledon":       "Grass",
    "US Open":         "Hard",
}
ROUND_ORDER = {"R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "SF": 6, "F": 7}


def nkey(name):
    return str(name).replace("_", " ").strip().lower()


def build_2025_rows():
    matches_path = os.path.join(CHARTING_REPO, "charting-m-matches.csv")
    stats_path   = os.path.join(CHARTING_REPO, "charting-m-stats-Overview.csv")
    players_path = os.path.join(ATP_REPO, "atp_players.csv")
    rankings_path= os.path.join(ATP_REPO, "atp_rankings_current.csv")

    matches = pd.read_csv(matches_path, low_memory=False)
    matches["Date"] = pd.to_datetime(
        matches["Date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    matches["year"] = matches["Date"].dt.year

    gs2025 = matches[
        (matches["year"] == 2025)
        & matches["Tournament"].str.contains(
            "Australian Open|Roland Garros|Wimbledon|US Open",
            case=False, na=False
        )
        & ~matches["Tournament"].str.contains("Junior", case=False, na=False)
        & ~matches["Round"].astype(str).str.startswith("Q")
    ].copy()

    print(f"2025 GS charted matches: {len(gs2025)}")

    players = pd.read_csv(players_path, low_memory=False)
    players["full_name"] = (
        players["name_first"].astype(str).str.strip()
        + " "
        + players["name_last"].astype(str).str.strip()
    ).str.lower()
    players["height"]    = pd.to_numeric(players["height"],    errors="coerce")
    players["player_id"] = players["player_id"].astype(str)
    name_to_ht   = dict(zip(players["full_name"], players["height"]))
    name_to_hand = dict(zip(players["full_name"], players["hand"]))

    rankings = pd.read_csv(
        rankings_path, header=None,
        names=["date", "rank", "player_id", "points"],
        low_memory=False,
    )
    rankings["date"]      = pd.to_datetime(
        rankings["date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    rankings["player_id"] = rankings["player_id"].astype(str)
    latest_rank = (
        rankings.sort_values("date")
        .groupby("player_id")["rank"]
        .last()
        .reset_index()
        .rename(columns={"rank": "atp_rank"})
    )
    players_ranked = players.merge(latest_rank, on="player_id", how="left")
    name_to_rank = dict(zip(players_ranked["full_name"], players_ranked["atp_rank"]))

    stats = pd.read_csv(stats_path, low_memory=False)
    stats_total = stats[stats["set"].astype(str).str.strip() == "Total"].copy()
    for c in ["bk_pts", "bp_saved"]:
        stats_total[c] = pd.to_numeric(stats_total[c], errors="coerce")

    rows = []
    for _, row in gs2025.iterrows():
        tournament = str(row["Tournament"])
        surface    = SLAM_SURF.get(tournament, str(row.get("Surface", "Hard")))
        rnd        = str(row["Round"])
        date       = row["Date"]
        winner_key = nkey(row["Player 1"])
        loser_key  = nkey(row["Player 2"])

        w_ht   = name_to_ht.get(winner_key, np.nan)
        l_ht   = name_to_ht.get(loser_key,  np.nan)
        w_hand = {"R":"R","L":"L","U":"Unknown"}.get(
            str(row.get("Pl 1 hand", "R")).strip(), "R"
        )
        l_hand = {"R":"R","L":"L","U":"Unknown"}.get(
            str(row.get("Pl 2 hand", "R")).strip(), "R"
        )
        w_rank = name_to_rank.get(winner_key, np.nan)
        l_rank = name_to_rank.get(loser_key,  np.nan)

        match_stats = stats_total[stats_total["match_id"] == row["match_id"]]
        w_bk=np.nan; w_bs=np.nan; l_bk=np.nan; l_bs=np.nan
        if len(match_stats) >= 2:
            w_bk = match_stats.iloc[0]["bk_pts"]
            w_bs = match_stats.iloc[0]["bp_saved"]
            l_bk = match_stats.iloc[1]["bk_pts"]
            l_bs = match_stats.iloc[1]["bp_saved"]

        w_bpFaced = (
            (w_bk + l_bs) if pd.notna(w_bk) and pd.notna(l_bs) else np.nan
        )
        l_bpFaced = (
            (l_bk + w_bs) if pd.notna(l_bk) and pd.notna(w_bs) else np.nan
        )

        rows.append({
            "tourney_id":    f"2025-{tournament.replace(' ', '_')}",
            "tourney_name":  tournament,
            "surface":       surface,
            "tourney_level": "G",
            "tourney_date":  date,
            "match_num":     ROUND_ORDER.get(rnd, 0) * 100,
            "winner_id":     np.nan,
            "winner_name":   winner_key,
            "winner_ht":     w_ht,
            "winner_hand":   w_hand,
            "winner_rank":   w_rank,
            "loser_id":      np.nan,
            "loser_name":    loser_key,
            "loser_ht":      l_ht,
            "loser_hand":    l_hand,
            "loser_rank":    l_rank,
            "score":         "",
            "round":         rnd,
            "minutes":       np.nan,
            "w_bpSaved":     w_bs,
            "w_bpFaced":     w_bpFaced,
            "l_bpSaved":     l_bs,
            "l_bpFaced":     l_bpFaced,
            "year":          2025,
        })

    df_2025 = pd.DataFrame(rows)
    df_2025["tourney_date"] = pd.to_datetime(df_2025["tourney_date"])
    return df_2025


def run():
    print("=" * 60)
    print("  Building atp_grand_slams_1990_2025.parquet")
    print("  Source: MatchChartingProject (2025) + existing parquet (1990-2024)")
    print("=" * 60)

    existing = pd.read_parquet(EXISTING_GS)
    df_2025  = build_2025_rows()

    num_cols = [
        "winner_ht", "loser_ht", "winner_rank", "loser_rank",
        "minutes", "w_bpFaced", "w_bpSaved", "l_bpFaced", "l_bpSaved",
        "match_num",
    ]
    for df in [existing, df_2025]:
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    all_cols = list(existing.columns)
    for c in all_cols:
        if c not in df_2025.columns:
            df_2025[c] = np.nan
    df_2025 = df_2025[all_cols]

    combined = pd.concat([existing, df_2025], ignore_index=True)
    combined["tourney_date"] = pd.to_datetime(combined["tourney_date"])
    combined["year"] = combined["tourney_date"].dt.year.astype(int)
    combined = combined.sort_values(
        ["tourney_date", "match_num"]
    ).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_PARQUET), exist_ok=True)
    combined.to_parquet(OUTPUT_PARQUET, index=False)

    print(f"\nSaved: {OUTPUT_PARQUET}")
    print(f"Total rows : {len(combined)}")
    print(f"Years      : {combined['year'].min()} - {combined['year'].max()}")
    print(f"2024 rows  : {(combined['year']==2024).sum()}")
    print(f"2025 rows  : {(combined['year']==2025).sum()}")

    print("\nNull rates in 2025 subset:")
    sub = combined[combined["year"] == 2025]
    for c in ["winner_name", "winner_ht", "winner_rank", "winner_hand", "w_bpFaced"]:
        print(f"  {c:<20}: {sub[c].isna().mean()*100:.0f}% null")

    print("\nDone. You can now run baseline.py and dynamic_elo.py.")


if __name__ == "__main__":
    run()