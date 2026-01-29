import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

APP_TITLE = "EWO_CARONTE_WLF — Signal Lab"
DEFAULT_DATA_DIR = "data"

SIGNAL_EVENTS = {"REVERSAL", "TREND", "CROSS_BAND", "BAND_BREAK"}  # extend if you add more
OPPOSITE_EVENT_POOL = {"REVERSAL", "TREND", "CROSS_BAND", "BAND_BREAK"}

FEATURE_COLS = [
    "barsSinceZoneEnd",
    "pullbackDepthPct",
    "reenteredZone",
    "signalBodyAtr",
    "maxBodyAtrLastM",
    "distFromEmaAtr",
    "efficiencyLastM",
    "deadzoneLen",
    "firstBreakAfterDeadzone",
]

DERIVED_FEATURE_COLS = [
    "ewoBandRatio",
    "sigGapBand",
    "rangeAtr",
    "bodyAtr",
]


def _safe_json_loads(line: str):
    try:
        return json.loads(line)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_jsonl_files(file_paths: list[str]) -> tuple[pd.DataFrame, dict]:
    rows = []
    bad = 0
    for fp in file_paths:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = _safe_json_loads(line)
                if obj is None:
                    bad += 1
                    continue
                obj["_file"] = os.path.basename(fp)
                rows.append(obj)

    if not rows:
        return pd.DataFrame(), {"files": len(file_paths), "rows": 0, "bad_lines": bad}

    df = pd.DataFrame(rows)

    # types
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    for c in ["bar"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["o","h","l","c","ewo","sig","bandAbs","atr"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # booleans
    for c in ["insideBand","reenteredZone","firstBreakAfterDeadzone"]:
        if c in df.columns:
            if df[c].dtype != bool:
                df[c] = df[c].astype(str).str.lower().isin(["true","1","t","yes","y"])

    # basic derived time cols
    if "ts" in df.columns:
        df["date"] = df["ts"].dt.date
        df["hour"] = df["ts"].dt.hour

    # ensure required columns exist
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = np.nan

    # derived features
    df["ewoBandRatio"] = np.where(df["bandAbs"] > 0, np.abs(df["ewo"]) / df["bandAbs"], np.nan)
    df["sigGapBand"]   = np.where(df["bandAbs"] > 0, np.abs(df["ewo"] - df["sig"]) / df["bandAbs"], np.nan)
    df["rangeAtr"]     = np.where(df["atr"] > 0, (df["h"] - df["l"]) / df["atr"], np.nan)
    df["bodyAtr"]      = np.where(df["atr"] > 0, np.abs(df["c"] - df["o"]) / df["atr"], np.nan)

    # normalize some numeric features
    for c in ["barsSinceZoneEnd","pullbackDepthPct","signalBodyAtr","maxBodyAtrLastM","distFromEmaAtr","efficiencyLastM","deadzoneLen"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    meta = {
        "files": len(file_paths),
        "rows": len(df),
        "bad_lines": bad,
        "unique_instruments": df["instrument"].nunique() if "instrument" in df.columns else 0,
    }
    return df, meta


def dir_sign(dir_val: str) -> int:
    if not isinstance(dir_val, str):
        return 0
    d = dir_val.strip().upper()
    if d in {"UP", "LONG", "BUY", "BULL"}:
        return +1
    if d in {"DOWN", "SHORT", "SELL", "BEAR"}:
        return -1
    return 0


def compute_pseudo_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Compute outcomes using *next opposite-direction signal event* close as a proxy.
    Not a true MFE/MAE, but useful for relative comparisons while you build richer logs."""
    if df.empty:
        return df

    df = df.sort_values(["instrument", "tf", "bar", "ts"]).reset_index(drop=True)
    df["dirSign"] = df["dir"].apply(dir_sign)

    is_signal = df["event"].isin(SIGNAL_EVENTS) & (df["dirSign"] != 0)
    df["isSignal"] = is_signal

    # Next opposite signal index per (instrument, tf)
    df["nextOppBar"] = np.nan
    df["nextOppC"] = np.nan
    df["barsToOpp"] = np.nan
    df["deltaToOppAtr"] = np.nan
    df["deltaToOppPts"] = np.nan

    df["nextDeadzoneBar"] = np.nan
    df["barsToDeadzone"] = np.nan

    for (inst, tf), g in df.groupby(["instrument", "tf"], dropna=False):
        idx = g.index.to_numpy()
        bars = g["bar"].to_numpy()
        closes = g["c"].to_numpy()
        atrs = g["atr"].to_numpy()
        dirs = g["dirSign"].to_numpy()
        events = g["event"].to_numpy()

        # Precompute positions of BAND_ENTER (deadzone start)
        band_enter_pos = np.where(events == "BAND_ENTER")[0]

        # Precompute positions by direction for signal events
        sig_pos_up = np.where((events.astype(object) != None) & np.isin(events, list(OPPOSITE_EVENT_POOL)) & (dirs == +1))[0]
        sig_pos_dn = np.where((events.astype(object) != None) & np.isin(events, list(OPPOSITE_EVENT_POOL)) & (dirs == -1))[0]

        # Helper to find next position > p in an array of positions
        def _next_pos(pos_arr, p):
            j = np.searchsorted(pos_arr, p + 1)
            if j >= len(pos_arr):
                return None
            return int(pos_arr[j])

        for local_i, global_i in enumerate(idx):
            if not is_signal.loc[global_i]:
                continue

            d = dirs[local_i]
            opp_arr = sig_pos_dn if d == +1 else sig_pos_up
            opp_p = _next_pos(opp_arr, local_i)
            if opp_p is not None:
                df.at[global_i, "nextOppBar"] = bars[opp_p]
                df.at[global_i, "nextOppC"] = closes[opp_p]
                df.at[global_i, "barsToOpp"] = bars[opp_p] - bars[local_i]

                delta_pts = (closes[opp_p] - closes[local_i]) * d
                df.at[global_i, "deltaToOppPts"] = delta_pts
                df.at[global_i, "deltaToOppAtr"] = (delta_pts / atrs[local_i]) if atrs[local_i] and atrs[local_i] > 0 else np.nan

            # time to next deadzone entry
            be_p = _next_pos(band_enter_pos, local_i)
            if be_p is not None:
                df.at[global_i, "nextDeadzoneBar"] = bars[be_p]
                df.at[global_i, "barsToDeadzone"] = bars[be_p] - bars[local_i]

    # Convenience label
    df["pseudoWin"] = np.where(df["isSignal"], df["deltaToOppAtr"] > 0, np.nan)

    return df


def kpi_row(df_sig: pd.DataFrame):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Signals", f"{len(df_sig):,}")
    c2.metric("Pseudo win rate", f"{(df_sig['pseudoWin'].mean()*100):.1f}%" if len(df_sig) else "—")
    c3.metric("Median Δ to opp (ATR)", f"{df_sig['deltaToOppAtr'].median():.2f}" if len(df_sig) else "—")
    c4.metric("Median bars to opp", f"{df_sig['barsToOpp'].median():.0f}" if len(df_sig) else "—")


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    # Data source
    st.sidebar.header("Data")
    data_dir = st.sidebar.text_input("Data folder", DEFAULT_DATA_DIR)
    folder_files = sorted([str(p) for p in Path(data_dir).glob("*.jsonl")]) if data_dir else []

    uploaded = st.sidebar.file_uploader("Upload .jsonl (optional)", type=["jsonl"], accept_multiple_files=True)

    file_paths = []
    temp_dir = None
    if uploaded:
        temp_dir = Path(st.session_state.get("_tmp_dir", ".streamlit_tmp"))
        temp_dir.mkdir(exist_ok=True)
        st.session_state["_tmp_dir"] = str(temp_dir)
        for uf in uploaded:
            out = temp_dir / uf.name
            out.write_bytes(uf.getvalue())
            file_paths.append(str(out))
    else:
        file_paths = folder_files

    if not file_paths:
        st.info("Add your JSONL files into ./data or upload them in the sidebar.")
        st.stop()

    df, meta = load_jsonl_files(file_paths)

    # Header KPIs
    a, b, c, d = st.columns(4)
    a.metric("Files", meta["files"])
    b.metric("Rows", f"{meta['rows']:,}")
    c.metric("Bad lines", meta["bad_lines"])
    d.metric("Unique instruments", meta["unique_instruments"])

    if df.empty:
        st.warning("No rows loaded.")
        st.stop()

    # Filters
    st.sidebar.header("Filters")
    insts = sorted(df["instrument"].dropna().unique().tolist())
    tfs = sorted(df["tf"].dropna().unique().tolist())
    evs = sorted(df["event"].dropna().unique().tolist())

    inst_sel = st.sidebar.multiselect("Instrument", insts, default=insts[:1] if insts else None)
    tf_sel = st.sidebar.multiselect("Timeframe", tfs, default=tfs[:1] if tfs else None)
    ev_sel = st.sidebar.multiselect("Event types", evs, default=evs)

    # Date range
    if "ts" in df.columns and df["ts"].notna().any():
        min_dt = df["ts"].min().to_pydatetime()
        max_dt = df["ts"].max().to_pydatetime()
        dt_from, dt_to = st.sidebar.date_input("Date range", value=(min_dt.date(), max_dt.date()))
    else:
        dt_from = dt_to = None

    # Apply filters
    f = df.copy()
    if inst_sel:
        f = f[f["instrument"].isin(inst_sel)]
    if tf_sel:
        f = f[f["tf"].isin(tf_sel)]
    if ev_sel:
        f = f[f["event"].isin(ev_sel)]
    if dt_from and dt_to and "ts" in f.columns:
        f = f[(f["ts"].dt.date >= dt_from) & (f["ts"].dt.date <= dt_to)]

    st.subheader("Filtered summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Filtered rows", f"{len(f):,}")
    c2.metric("Days", f"{f['date'].nunique() if 'date' in f.columns else 0}")
    c3.metric("Event types", f"{f['event'].nunique()}")

    # Compute outcomes
    f2 = compute_pseudo_outcomes(f)

    # Focus: signal events
    sig_only = st.checkbox("Show only signal events (REVERSAL/TREND/CROSS_BAND/BAND_BREAK)", value=True)
    if sig_only:
        f2_view = f2[f2["isSignal"]].copy()
    else:
        f2_view = f2.copy()

    # KPIs for signals
    st.subheader("Signal quality (proxy)")
    kpi_row(f2_view)

    # Events per day chart
    if "date" in f2.columns:
        per_day = f2.groupby(["date", "event"], as_index=False).size()
        chart = (
            alt.Chart(per_day)
            .mark_line()
            .encode(
                x=alt.X("date:T", title="Day"),
                y=alt.Y("size:Q", title="Events"),
                color=alt.Color("event:N", title="Event"),
                tooltip=["date:T", "event:N", "size:Q"],
            )
            .properties(height=260)
        )
        st.altair_chart(chart, use_container_width=True)

    # Feature explorer
    st.subheader("Feature explorer")
    left, right = st.columns([1, 1])

    with left:
        feat = st.selectbox("Feature", options=[c for c in FEATURE_COLS + DERIVED_FEATURE_COLS if c in f2_view.columns], index=0)
        yopt = st.selectbox("Outcome", options=["deltaToOppAtr", "barsToOpp", "barsToDeadzone"], index=0)

        base = f2_view[[feat, yopt, "event", "dir", "instrument", "tf"]].dropna()
        if base.empty:
            st.info("Not enough data after filtering.")
        else:
            scatter = (
                alt.Chart(base)
                .mark_circle(size=45, opacity=0.35)
                .encode(
                    x=alt.X(f"{feat}:Q", title=feat),
                    y=alt.Y(f"{yopt}:Q", title=yopt),
                    color=alt.Color("event:N", legend=None),
                    tooltip=["instrument", "tf", "event", "dir", feat, yopt],
                )
                .properties(height=320)
            )
            st.altair_chart(scatter, use_container_width=True)

    with right:
        # Quantile box plot
        base2 = f2_view[[feat, "deltaToOppAtr"]].dropna()
        if len(base2) >= 20:
            base2 = base2.copy()
            base2["q"] = pd.qcut(base2[feat], 5, duplicates="drop")
            box = (
                alt.Chart(base2)
                .mark_boxplot()
                .encode(
                    x=alt.X("q:N", title=f"{feat} (quintiles)"),
                    y=alt.Y("deltaToOppAtr:Q", title="Δ to opp (ATR)"),
                )
                .properties(height=320)
            )
            st.altair_chart(box, use_container_width=True)
        else:
            st.info("Need at least ~20 signal rows to show quantile box plot.")

    # Correlation table
    st.subheader("Feature correlations (signals only)")
    corr_cols = [c for c in (FEATURE_COLS + DERIVED_FEATURE_COLS + ["deltaToOppAtr", "barsToOpp"]) if c in f2_view.columns]
    corr_df = f2_view[corr_cols].copy()

    # Convert booleans to 0/1 for corr
    for c in ["reenteredZone","firstBreakAfterDeadzone"]:
        if c in corr_df.columns:
            corr_df[c] = corr_df[c].astype(int)

    corr = corr_df.corr(numeric_only=True)
    st.dataframe(corr.style.format("{:.2f}"), use_container_width=True)

    # Rule builder
    st.subheader("A+ rule builder (interactive)")
    st.caption("This uses the proxy outcome Δ-to-next-opposite-signal (in ATR). It’s not perfect, but it’s enough to discover which filters improve expectancy.")

    controls = st.columns(4)
    min_strength = controls[0].slider("Min ewoBandRatio", 0.0, float(np.nanmax(f2_view["ewoBandRatio"].values) if "ewoBandRatio" in f2_view else 5.0), 1.0, 0.1)
    max_body = controls[1].slider("Max signalBodyAtr", 0.0, float(np.nanmax(f2_view["signalBodyAtr"].values) if "signalBodyAtr" in f2_view else 10.0), 3.0, 0.1)
    max_deadzone = controls[2].slider("Max deadzoneLen", 0, int(np.nanmax(f2_view["deadzoneLen"].values) if "deadzoneLen" in f2_view else 200), 80, 1)
    need_first_break = controls[3].checkbox("Require firstBreakAfterDeadzone", value=False)

    filtered = f2_view.copy()
    if "ewoBandRatio" in filtered.columns:
        filtered = filtered[filtered["ewoBandRatio"] >= min_strength]
    if "signalBodyAtr" in filtered.columns:
        filtered = filtered[filtered["signalBodyAtr"] <= max_body]
    if "deadzoneLen" in filtered.columns:
        filtered = filtered[filtered["deadzoneLen"] <= max_deadzone]
    if need_first_break and "firstBreakAfterDeadzone" in filtered.columns:
        filtered = filtered[filtered["firstBreakAfterDeadzone"] == True]

    kpi_row(filtered)

    st.write("Top signals (by Δ-to-opp ATR)")
    cols_show = ["ts","instrument","tf","event","dir","bar","c","atr","ewo","sig","bandAbs","ewoBandRatio","signalBodyAtr","deadzoneLen","barsSinceZoneEnd","barsToOpp","deltaToOppAtr"]
    cols_show = [c for c in cols_show if c in filtered.columns]
    st.dataframe(filtered.sort_values("deltaToOppAtr", ascending=False).head(50)[cols_show], use_container_width=True)

    st.write("Table preview")
    cols_preview = ["ts","instrument","tf","event","dir","bar","o","h","l","c","ewo","sig","insideBand","bandAbs","atr"]
    cols_preview = [c for c in cols_preview if c in f2.columns]
    st.dataframe(f2.head(200)[cols_preview], use_container_width=True)


if __name__ == "__main__":
    main()
