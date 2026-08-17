"""
=============================================================================
MODULE 01 — DATA INGESTION & CLEANING  (GitHub-direct version)
Tennis Grand Slam Winner Prediction | Dynamic ELO Pipeline
=============================================================================
Strategy : git clone --depth=1 both Sackmann repos into /tmp at runtime.
           No manual downloads needed. Re-run anytime to get latest data.

Sources  :
  tennis_atp               → atp_matches_YYYY.csv  (match-level, 1968–2026)
  tennis_MatchChartingProject → charting-m-matches.csv
                              → charting-m-stats-Overview.csv  (bp stats)
=============================================================================
"""

import os
import re
import subprocess
import logging

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SECTION 1 — CONFIGURATION
# ---------------------------------------------------------------------------

# ── GitHub repo URLs ───────────────────────────────────────────────────────
ATP_REPO_URL      = "https://github.com/JeffSackmann/tennis_atp.git"
CHARTING_REPO_URL = "https://github.com/JeffSackmann/tennis_MatchChartingProject.git"

# ── Local clone destinations ───────────────────────────────────────────────
ATP_CLONE_DIR      = "/tmp/tennis_atp"
CHARTING_CLONE_DIR = "/tmp/tennis_charting"

# ── Output directory for cleaned Parquet files ────────────────────────────
OUTPUT_DIR = "data/cleaned"

# ── Year range to load from the ATP repo ──────────────────────────────────
ATP_YEAR_START = 1990   # older data → better ELO convergence
ATP_YEAR_END   = 2026

# ── ATP columns we actually need (verified against real repo) ─────────────
ATP_COLUMNS_NEEDED = [
    "tourney_id",
    "tourney_name",
    "surface",
    "tourney_level",     # 'G' = Grand Slam
    "tourney_date",      # YYYYMMDD integer
    "match_num",
    "winner_id",
    "winner_name",
    "winner_ht",
    "winner_hand",
    "winner_rank",
    "loser_id",
    "loser_name",
    "loser_ht",
    "loser_hand",
    "loser_rank",
    "score",
    "round",
    "minutes",
    "w_bpSaved",
    "w_bpFaced",
    "l_bpSaved",
    "l_bpFaced",
]

# ── Charting column renames (spaced headers → snake_case) ─────────────────
CHARTING_MATCHES_RENAME = {
    "Player 1":   "player1",
    "Player 2":   "player2",
    "Pl 1 hand":  "p1_hand",
    "Pl 2 hand":  "p2_hand",
    "Date":       "charting_date",
    "Tournament": "tournament",
    "Time":       "duration_raw",
}

# ── Charting stats columns needed for Clutch Factor ───────────────────────
CHARTING_STATS_COLS = [
    "match_id",
    "player",
    "set",
    "bk_pts",    # break points faced (as returner)
    "bp_saved",  # break points saved (as server)
]


# ---------------------------------------------------------------------------
# SECTION 2 — GIT CLONE HELPER
# ---------------------------------------------------------------------------

def clone_repo(repo_url: str, clone_dir: str) -> None:
    """
    Shallow-clone a GitHub repo into clone_dir.
    If the directory already exists, pulls latest instead (idempotent).
    Uses --depth=1 so we skip full git history (much faster).
    """
    if os.path.exists(os.path.join(clone_dir, ".git")):
        log.info("Repo already cloned at %s — pulling latest...", clone_dir)
        result = subprocess.run(
            ["git", "-C", clone_dir, "pull", "--quiet"],
            capture_output=True, text=True,
        )
    else:
        log.info("Cloning %s → %s", repo_url, clone_dir)
        result = subprocess.run(
            [
                "git", "clone",
                "--depth=1",
                "--filter=blob:none",
                repo_url,
                clone_dir,
            ],
            capture_output=True, text=True,
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"Git operation failed for {repo_url}:\n{result.stderr}"
        )
    log.info("  ✓ Repo ready at %s", clone_dir)


# ---------------------------------------------------------------------------
# SECTION 3 — ATP DATA LOADER
# ---------------------------------------------------------------------------

def load_atp_matches(
    clone_dir: str  = ATP_CLONE_DIR,
    year_start: int = ATP_YEAR_START,
    year_end: int   = ATP_YEAR_END,
    grand_slams_only: bool = True,
) -> pd.DataFrame:
    """
    Read all atp_matches_YYYY.csv files in [year_start, year_end]
    from the cloned repo and concatenate them.

    Returns
    -------
    pd.DataFrame — raw (uncleaned) ATP match rows
    """
    log.info("━━━ Loading ATP match files (%d–%d) ━━━", year_start, year_end)
    frames = []

    for year in range(year_start, year_end + 1):
        filepath = os.path.join(clone_dir, f"atp_matches_{year}.csv")
        if not os.path.exists(filepath):
            continue  # some years simply don't have a file

        df_year = pd.read_csv(filepath, low_memory=False)
        frames.append(df_year)
        log.info("  %d → %d rows", year, len(df_year))

    if not frames:
        raise FileNotFoundError(
            f"No atp_matches_YYYY.csv files found in '{clone_dir}'."
        )

    df = pd.concat(frames, ignore_index=True)
    log.info("Combined rows before filtering: %d", len(df))

    # Pad any columns that are missing from older CSVs
    for col in ATP_COLUMNS_NEEDED:
        if col not in df.columns:
            log.warning("  Column '%s' missing — filling with NaN", col)
            df[col] = np.nan

    df = df[ATP_COLUMNS_NEEDED].copy()

    # Grand Slam filter
    if grand_slams_only:
        before = len(df)
        df = df[df["tourney_level"] == "G"].copy()
        log.info("Grand Slam filter: %d → %d rows", before, len(df))

    return df


# ---------------------------------------------------------------------------
# SECTION 4 — CHARTING DATA LOADER
# ---------------------------------------------------------------------------

def load_charting_data(
    clone_dir: str = CHARTING_CLONE_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load two charting files:
      1. charting-m-matches.csv       — match metadata (players, date, handedness)
      2. charting-m-stats-Overview.csv — per-player match totals (break points)

    Returns
    -------
    (df_matches, df_stats) — two raw DataFrames
    """
    log.info("━━━ Loading MatchChartingProject files ━━━")

    matches_path = os.path.join(clone_dir, "charting-m-matches.csv")
    stats_path   = os.path.join(clone_dir, "charting-m-stats-Overview.csv")

    for path in [matches_path, stats_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected charting file not found: {path}")

    df_matches = pd.read_csv(matches_path, low_memory=False)
    df_stats   = pd.read_csv(stats_path,   low_memory=False)

    log.info("  charting-m-matches.csv       → %d rows", len(df_matches))
    log.info("  charting-m-stats-Overview    → %d rows", len(df_stats))

    return df_matches, df_stats


# ---------------------------------------------------------------------------
# SECTION 5 — CLEANING: ATP DATA
# ---------------------------------------------------------------------------

def clean_atp_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise and clean the raw ATP DataFrame.

    Steps
    -----
    1. Parse tourney_date → datetime + extract 'year'
    2. Normalise player names → lowercase, stripped
    3. Normalise handedness   → 'R' / 'L' / 'Unknown'
    4. Cast all numeric columns safely
    5. Drop rows missing winner / loser / date
    6. Drop walkovers, retirements, defaults
    7. Sort chronologically
    """
    log.info("━━━ Cleaning ATP data ━━━")
    df = df.copy()

    # 1. Parse date
    df["tourney_date"] = pd.to_datetime(
        df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    df["year"] = df["tourney_date"].dt.year
    log.info("  Null dates: %d", df["tourney_date"].isna().sum())

    # 2. Normalise player names
    # CRITICAL: must match the same normalisation used on charting data
    for col in ["winner_name", "loser_name"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.lower()
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

    # 3. Normalise handedness
    hand_map = {"R": "R", "L": "L", "U": "Unknown", "": "Unknown", "nan": "Unknown"}
    for col in ["winner_hand", "loser_hand"]:
        df[col] = (
            df[col].astype(str).str.strip()
            .map(hand_map)
            .fillna("Unknown")
        )

    # 4. Cast numerics
    numeric_cols = [
        "winner_ht", "loser_ht",
        "winner_rank", "loser_rank",
        "minutes",
        "w_bpSaved", "w_bpFaced",
        "l_bpSaved", "l_bpFaced",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5. Drop unplayable rows
    before = len(df)
    df = df.dropna(subset=["winner_name", "loser_name", "tourney_date"]).copy()
    log.info("  Dropped %d rows missing winner/loser/date", before - len(df))

    # 6. Drop walkovers / retirements / defaults
    before = len(df)
    df = df[
        ~df["score"].astype(str).str.contains(r"W/O|RET|DEF", na=False)
    ].copy()
    log.info("  Dropped %d walkovers/retirements/defaults", before - len(df))

    # 7. Sort chronologically
    df = df.sort_values(
        ["tourney_date", "match_num"], ascending=True
    ).reset_index(drop=True)

    log.info("  ✓ Clean ATP shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# SECTION 6 — CLEANING: CHARTING DATA
# ---------------------------------------------------------------------------

def clean_charting_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean charting-m-matches.csv.

    Steps
    -----
    1. Rename spaced headers to snake_case
    2. Parse charting_date → datetime
    3. Normalise player names (same convention as ATP)
    4. Normalise handedness
    5. Parse duration string "H:MM" → total minutes (float)
    """
    log.info("━━━ Cleaning charting matches ━━━")
    df = df.copy()

    # 1. Rename headers
    df = df.rename(columns=CHARTING_MATCHES_RENAME)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # 2. Parse date
    df["charting_date"] = pd.to_datetime(
        df["charting_date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    df["year"] = df["charting_date"].dt.year

    # 3. Normalise player names
    for col in ["player1", "player2"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.lower()
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

    # 4. Normalise handedness
    hand_map = {"R": "R", "L": "L", "U": "Unknown", "": "Unknown", "nan": "Unknown"}
    for col in ["p1_hand", "p2_hand"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().map(hand_map).fillna("Unknown")

    # 5. Parse duration "H:MM" → float minutes
    def parse_duration(val: str) -> float:
        val = str(val).strip()
        if re.match(r"^\d+:\d+$", val):
            h, m = val.split(":")
            return int(h) * 60 + int(m)
        return np.nan

    if "duration_raw" in df.columns:
        df["charting_minutes"] = df["duration_raw"].apply(parse_duration)
    else:
        df["charting_minutes"] = np.nan

    log.info("  ✓ Clean charting matches shape: %s", df.shape)
    return df


def clean_charting_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean charting-m-stats-Overview.csv.

    Keeps only set == 'Total' rows (one row per player per match),
    then casts break point columns to numeric.
    These feed into the Clutch Factor feature in Module 03.
    """
    log.info("━━━ Cleaning charting stats ━━━")
    df = df.copy()

    # Keep only the full-match aggregate row per player
    before = len(df)
    df = df[df["set"].astype(str).str.strip() == "Total"].copy()
    log.info("  set='Total' filter: %d → %d rows", before, len(df))

    # Keep only needed columns
    keep = [c for c in CHARTING_STATS_COLS if c in df.columns]
    df = df[keep].copy()

    # Cast numerics
    for col in ["bk_pts", "bp_saved"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalise player name
    df["player"] = (
        df["player"]
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    log.info("  ✓ Clean charting stats shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# SECTION 7 — DATA QUALITY REPORT
# ---------------------------------------------------------------------------

def print_quality_report(df: pd.DataFrame, label: str) -> None:
    """Concise quality snapshot — run after every load/clean step."""
    print(f"\n{'═' * 62}")
    print(f"  QUALITY REPORT — {label}")
    print(f"{'═' * 62}")
    print(f"  Shape       : {df.shape[0]:,} rows × {df.shape[1]} columns")

    for col in ["tourney_date", "charting_date"]:
        if col in df.columns and df[col].notna().any():
            print(f"  Date range  : {df[col].min().date()} → {df[col].max().date()}")
            break

    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if null_counts.empty:
        print("  Nulls       : None ✓")
    else:
        print(f"  Null columns: {len(null_counts)}")
        for col, cnt in null_counts.items():
            pct = cnt / len(df) * 100
            print(f"    • {col:<22} {cnt:>7,}  ({pct:.1f}%)")

    if "year" in df.columns:
        print(f"\n  Rows per year (most recent 8):")
        vc = df["year"].value_counts().sort_index().tail(8)
        for yr, cnt in vc.items():
            print(f"    {int(yr)}  →  {cnt:,}")

    print(f"{'═' * 62}\n")


# ---------------------------------------------------------------------------
# SECTION 8 — SAVE CLEANED DATA
# ---------------------------------------------------------------------------

def save_cleaned(df: pd.DataFrame, filename: str, output_dir: str = OUTPUT_DIR) -> None:
    """Save cleaned DataFrame to Parquet (fast, typed, compressed)."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    df.to_parquet(path, index=False)
    log.info("Saved → %s  (%d rows)", path, len(df))


# ---------------------------------------------------------------------------
# SECTION 9 — MAIN ORCHESTRATOR
# ---------------------------------------------------------------------------

def run_ingestion(
    atp_clone_dir: str      = ATP_CLONE_DIR,
    charting_clone_dir: str = CHARTING_CLONE_DIR,
    out_dir: str            = OUTPUT_DIR,
    year_start: int         = ATP_YEAR_START,
    year_end: int           = ATP_YEAR_END,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full ingestion pipeline:
      1. Clone / update both GitHub repos
      2. Load raw data
      3. Clean each dataset
      4. Print quality reports
      5. Save to Parquet

    Returns
    -------
    atp_clean        : Grand Slam match records, cleaned
    charting_matches : Charting match metadata, cleaned
    charting_stats   : Charting break point stats, cleaned
    """

    # Step 1 — Clone repos
    clone_repo(ATP_REPO_URL,      atp_clone_dir)
    clone_repo(CHARTING_REPO_URL, charting_clone_dir)

    # Step 2 & 3 — ATP
    atp_raw   = load_atp_matches(atp_clone_dir, year_start, year_end)
    atp_clean = clean_atp_data(atp_raw)
    print_quality_report(atp_clean, "ATP Grand Slam Matches")
    save_cleaned(atp_clean, "atp_grand_slams_clean.parquet", out_dir)

    # Step 2 & 3 — Charting
    raw_matches, raw_stats = load_charting_data(charting_clone_dir)
    charting_matches = clean_charting_matches(raw_matches)
    charting_stats   = clean_charting_stats(raw_stats)

    print_quality_report(charting_matches, "Charting Matches")
    print_quality_report(charting_stats,   "Charting Stats (Break Points)")

    save_cleaned(charting_matches, "charting_matches_clean.parquet", out_dir)
    save_cleaned(charting_stats,   "charting_stats_clean.parquet",   out_dir)

    log.info("✓ All ingestion complete. Files saved to '%s'.", out_dir)
    return atp_clean, charting_matches, charting_stats


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    atp_df, chart_matches_df, chart_stats_df = run_ingestion()