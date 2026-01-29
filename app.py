import json
import os
from pathlib import Path
from typing import Optional, Tuple

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


# ==========================================================
# Candle loader (NinjaTrader "Last" TXT export)
# Example line:
# 20251103 000200;26031.25;26049;26029.25;26047.75;1637
# (date) (time);O;H;L;C;V
# ==========================================================
def load_candles_last_txt(file_path: str) -> pd.DataFrame:
    rows = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # Some exports might use comma as decimal in other locales; we normalize later.
            try:
                left, o, h, l, c, v = line.split(";")
                d, t = left.split()
                ts = pd.to_datetime(d + t, format="%Y%m%d%H%M%S", errors="raise")
                rows.append(
                    (
                        ts,
                        float(o.replace(",", ".")),
                        float(h.replace(",", ".")),
                        float(l.replace(",", ".")),
                        float(c.replace(",", ".")),
                        float(v.replace(",", ".")),
                    )
                )
            except Exception:
                # ignore malformed lines
                continue

    df = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c", "v"])
    if df.empty:
        return df
    df = df.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
    return df


def _nearest_merge_signals_to_candles(signals: pd.DataFrame, candles: pd.DataFrame, tol_seconds: int = 60) -> pd.DataFrame:
    """Merge on timestamp; if exact match missing, use nearest within tolerance."""
    if signals.empty or candles.empty:
        return signals

    s = signals.copy().sort_values("ts")
    c = candles.copy().sort_values("ts")

    merged = pd.merge_asof(
        s,
        c,
        on="ts",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=tol_seconds),
        suffixes=("", "_px"),
    )
    return merged


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


def compute_true_outcomes_from_candles(
    df: pd.DataFrame,
    candles: pd.DataFrame,
    horizon_bars: int = 30,
    match_tolerance_sec: int = 60,
    good_mfe_atr: float = 1.0,
    bad_mae_atr: float = 1.0,
) -> pd.DataFrame:
    """Compute forward MFE/MAE outcomes using candle series.

    - Matches df.ts to nearest candle ts within tolerance.
    - Uses entry = candle close on the signal bar.
    - Direction from df.dir (UP/DOWN).
    - Normalizes by df.atr if present/positive, else uses candle-based ATR(14).

    Adds:
      hasCandleMatch, candleIdx, mfeAtr, maeAtr, endAtr, goodSignal
    """
    if df.empty or candles is None or candles.empty:
        out = df.copy()
        for col in ["hasCandleMatch", "candleIdx", "mfeAtr", "maeAtr", "endAtr", "goodSignal"]:
            if col not in out.columns:
                out[col] = np.nan
        return out

    out = df.copy()
    if "ts" not in out.columns:
        out["hasCandleMatch"] = False
        return out

    # Ensure candle columns
    for req in ["ts", "h", "l", "c"]:
        if req not in candles.columns:
            out["hasCandleMatch"] = False
            return out

    candles = candles.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    c_ts = candles["ts"].values.astype("datetime64[ns]")
    s_ts = pd.to_datetime(out["ts"], errors="coerce").values.astype("datetime64[ns]")

    # Nearest index via searchsorted
    pos = np.searchsorted(c_ts, s_ts)
    pos = np.clip(pos, 0, len(c_ts) - 1)
    # choose closer of pos and pos-1
    pos_prev = np.clip(pos - 1, 0, len(c_ts) - 1)
    ts_pos = c_ts[pos]
    ts_prev = c_ts[pos_prev]
    diff_pos = np.abs((ts_pos - s_ts).astype("timedelta64[s]").astype(int))
    diff_prev = np.abs((s_ts - ts_prev).astype("timedelta64[s]").astype(int))
    use_prev = diff_prev <= diff_pos
    idx = np.where(use_prev, pos_prev, pos)

    nearest = c_ts[idx]
    diff = np.abs((nearest - s_ts).astype("timedelta64[s]").astype(int))
    out["candleIdx"] = idx
    out["hasCandleMatch"] = diff <= int(match_tolerance_sec)

    # Candle-based ATR(14) fallback
    if "atr" not in out.columns:
        out["atr"] = np.nan
    if out["atr"].isna().any() or (out["atr"] <= 0).any():
        prev_c = candles["c"].shift(1)
        tr = pd.concat(
            [
                (candles["h"] - candles["l"]).abs(),
                (candles["h"] - prev_c).abs(),
                (candles["l"] - prev_c).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr14 = tr.rolling(14, min_periods=14).mean()
    else:
        atr14 = None

    h = candles["h"].to_numpy(dtype=float)
    l = candles["l"].to_numpy(dtype=float)
    c = candles["c"].to_numpy(dtype=float)
    atr_fallback = atr14.to_numpy(dtype=float) if atr14 is not None else None

    mfe = np.full(len(out), np.nan)
    mae = np.full(len(out), np.nan)
    endr = np.full(len(out), np.nan)

    dir_sign = out.get("dir", pd.Series(index=out.index, dtype=object)).map({"UP": 1, "DOWN": -1})
    dir_sign = dir_sign.fillna(0).to_numpy(dtype=int)

    for k in range(len(out)):
        if not bool(out.loc[out.index[k], "hasCandleMatch"]):
            continue
        d = int(dir_sign[k])
        if d == 0:
            continue
        i = int(idx[k])
        j_end = min(len(c) - 1, i + int(max(1, horizon_bars)))
        if j_end <= i:
            continue

        atrv = float(out.loc[out.index[k], "atr"]) if "atr" in out.columns else np.nan
        if not np.isfinite(atrv) or atrv <= 0:
            atrv = float(atr_fallback[i]) if atr_fallback is not None and np.isfinite(atr_fallback[i]) else np.nan
        if not np.isfinite(atrv) or atrv <= 0:
            continue

        entry = float(c[i])
        fut_h = h[i + 1 : j_end + 1]
        fut_l = l[i + 1 : j_end + 1]
        if fut_h.size == 0:
            continue

        if d == 1:
            mfe_pts = max(0.0, float(np.max(fut_h - entry)))
            mae_pts = max(0.0, float(np.max(entry - fut_l)))
            end_pts = float(c[j_end] - entry)
        else:
            mfe_pts = max(0.0, float(np.max(entry - fut_l)))
            mae_pts = max(0.0, float(np.max(fut_h - entry)))
            end_pts = float(entry - c[j_end])

        mfe[k] = mfe_pts / atrv
        mae[k] = mae_pts / atrv
        endr[k] = end_pts / atrv

    out["mfeAtr"] = mfe
    out["maeAtr"] = mae
    out["endAtr"] = endr
    out["goodSignal"] = (out["mfeAtr"] >= float(good_mfe_atr)) & (out["maeAtr"] <= float(bad_mae_atr))
    return out


def kpi_row(df_sig: pd.DataFrame):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Signals", f"{len(df_sig):,}")
    if len(df_sig) == 0:
        c2.metric("Win rate", "—")
        c3.metric("Median outcome", "—")
        c4.metric("Median risk", "—")
        return

    if "goodSignal" in df_sig.columns:
        c2.metric("Win rate (true)", f"{(df_sig['goodSignal'].mean()*100):.1f}%")
        c3.metric("Median MFE (ATR)", f"{df_sig['mfeAtr'].median():.2f}")
        c4.metric("Median MAE (ATR)", f"{df_sig['maeAtr'].median():.2f}")
    else:
        c2.metric("Pseudo win rate", f"{(df_sig['pseudoWin'].mean()*100):.1f}%")
        c3.metric("Median Δ to opp (ATR)", f"{df_sig['deltaToOppAtr'].median():.2f}")
        c4.metric("Median bars to opp", f"{df_sig['barsToOpp'].median():.0f}")


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    # Data source
    st.sidebar.header("Data")
    data_dir = st.sidebar.text_input("Data folder", DEFAULT_DATA_DIR)
    folder_files = sorted([str(p) for p in Path(data_dir).glob("*.jsonl")]) if data_dir else []

    uploaded = st.sidebar.file_uploader("Upload .jsonl (optional)", type=["jsonl"], accept_multiple_files=True)

    st.sidebar.header("Candles (optional)")
    candles_up = st.sidebar.file_uploader(
        "Upload NinjaTrader Last export (.txt)",
        type=["txt"],
        accept_multiple_files=False,
        help="If provided, the app computes real forward MFE/MAE outcomes instead of proxy outcomes.",
    )
    with st.sidebar.expander("Outcome settings", expanded=False):
        horizon_bars = st.slider("Forward horizon (bars)", 5, 300, 30, 5)
        match_tol = st.slider("Timestamp match tolerance (sec)", 0, 180, 60, 10)
        good_mfe = st.slider("Good signal threshold (MFE in ATR)", 0.25, 5.0, 1.0, 0.25)
        bad_mae = st.slider("Max allowed MAE (ATR)", 0.25, 5.0, 1.0, 0.25)

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

    # ----------------------------------
    # Optional candles (for true outcomes)
    # ----------------------------------
    candles: Optional[pd.DataFrame] = None
    if candles_up is not None:
        try:
            ck = f"__candles__{candles_up.name}__{candles_up.size}"
            if ck in st.session_state:
                candles = st.session_state[ck]
            else:
                # Save to temp so the existing loader can read it
                ctmp = Path(st.session_state.get("_tmp_dir", ".")) / f"_candles_{candles_up.name}"
                ctmp.write_bytes(candles_up.getvalue())
                candles = load_candles_last_txt(str(ctmp))
                st.session_state[ck] = candles
        except Exception as ex:
            st.sidebar.warning(f"Could not parse candle file: {ex}")
            candles = None

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

    # If we have candles, compute real forward outcomes
    if candles is not None and not f2_view.empty:
        f2_view = compute_true_outcomes_from_candles(
            f2_view,
            candles,
            horizon_bars=horizon_bars,
            match_tolerance_sec=match_tolerance_sec,
            good_mfe_atr=good_mfe_atr,
            bad_mae_atr=bad_mae_atr,
        )

    # KPIs for signals
    st.subheader("Signal quality" + (" (true outcomes)" if candles is not None else " (proxy)"))
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
        outcome_opts = ["deltaToOppAtr", "barsToOpp", "barsToDeadzone"]
        if candles is not None:
            outcome_opts += ["mfeAtr", "maeAtr", "endAtr", "goodSignal"]
        yopt = st.selectbox("Outcome", options=outcome_opts, index=0)

        base = f2_view[[feat, yopt, "event", "dir", "instrument", "tf"]].dropna()
        if base.empty:
            st.info("Not enough data after filtering.")
        else:
            if yopt == "goodSignal":
                binned = base.copy()
                # Bin X into 20 buckets and plot mean success rate
                binned["xbin"] = pd.cut(binned[feat], bins=20)
                agg = binned.groupby(["xbin"], as_index=False).agg(good_rate=(yopt, "mean"), count=(yopt, "size"))
                agg["xmid"] = agg["xbin"].apply(lambda r: (r.left + r.right) / 2 if pd.notnull(r) else np.nan)

                line = (
                    alt.Chart(agg.dropna())
                    .mark_line()
                    .encode(
                        x=alt.X("xmid:Q", title=feat),
                        y=alt.Y("good_rate:Q", title=f"P(good) | horizon={horizon_bars} bars"),
                        tooltip=["xmid:Q", "good_rate:Q", "count:Q"],
                    )
                    .properties(height=320)
                )
                st.altair_chart(line, use_container_width=True)
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
        # Quantile summary
        base2 = f2_view[[feat, yopt]].dropna()
        if len(base2) >= 20 and yopt != "goodSignal":
            base2 = base2.copy()
            base2["q"] = pd.qcut(base2[feat], 5, duplicates="drop")
            box = (
                alt.Chart(base2)
                .mark_boxplot()
                .encode(
                    x=alt.X("q:N", title=f"{feat} (quintiles)"),
                    y=alt.Y(f"{yopt}:Q", title=yopt),
                )
                .properties(height=320)
            )
            st.altair_chart(box, use_container_width=True)
        elif yopt == "goodSignal" and len(base2) >= 20:
            base2 = base2.copy()
            base2["q"] = pd.qcut(base2[feat], 5, duplicates="drop")
            agg = base2.groupby("q", as_index=False).agg(good_rate=(yopt, "mean"), count=(yopt, "size"))
            bar = (
                alt.Chart(agg)
                .mark_bar(opacity=0.8)
                .encode(
                    x=alt.X("q:N", title=f"{feat} (quintiles)"),
                    y=alt.Y("good_rate:Q", title="P(good)"),
                    tooltip=["q:N", "good_rate:Q", "count:Q"],
                )
                .properties(height=320)
            )
            st.altair_chart(bar, use_container_width=True)
        else:
            st.info("Need at least ~20 signal rows to show this view.")

    # Correlation table
    st.subheader("Feature correlations (signals only)")
    extra_out_cols = ["deltaToOppAtr", "barsToOpp"]
    if candles is not None:
        extra_out_cols += ["mfeAtr", "maeAtr", "endAtr", "goodSignal"]
    corr_cols = [c for c in (FEATURE_COLS + DERIVED_FEATURE_COLS + extra_out_cols) if c in f2_view.columns]
    corr_df = f2_view[corr_cols].copy()

    # Convert booleans to 0/1 for corr
    for c in ["reenteredZone","firstBreakAfterDeadzone"]:
        if c in corr_df.columns:
            corr_df[c] = corr_df[c].astype(int)

    corr = corr_df.corr(numeric_only=True)
    st.dataframe(corr.style.format("{:.2f}"), use_container_width=True)

    # Rule builder
    st.subheader("A+ rule builder (interactive)")
    if candles is not None:
        st.caption(f"This uses *real* forward outcomes computed from candles (horizon: {horizon_bars} bars). Use it to find filters that raise MFE and reduce MAE.")
    else:
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

    if candles is not None and "mfeAtr" in filtered.columns:
        st.write(f"Top signals (by MFE over next {horizon_bars} bars, ATR units)")
        sort_col = "mfeAtr"
        cols_show = [
            "ts","instrument","tf","event","dir","bar","c","atr",
            "mfeAtr","maeAtr","endAtr","goodSignal",
            "ewo","sig","bandAbs","ewoBandRatio","signalBodyAtr","deadzoneLen",
            "distFromEmaAtr","efficiencyLastM","maxBodyAtrLastM","firstBreakAfterDeadzone",
        ]
    else:
        st.write("Top signals (by Δ-to-opp ATR) [proxy]")
        sort_col = "deltaToOppAtr"
        cols_show = ["ts","instrument","tf","event","dir","bar","c","atr","ewo","sig","bandAbs","ewoBandRatio","signalBodyAtr","deadzoneLen","barsSinceZoneEnd","barsToOpp","deltaToOppAtr"]
    cols_show = [c for c in cols_show if c in filtered.columns]
    st.dataframe(filtered.sort_values(sort_col, ascending=False).head(50)[cols_show], use_container_width=True)

    st.write("Table preview")
    cols_preview = ["ts","instrument","tf","event","dir","bar","o","h","l","c","ewo","sig","insideBand","bandAbs","atr"]
    cols_preview = [c for c in cols_preview if c in f2.columns]
    st.dataframe(f2.head(200)[cols_preview], use_container_width=True)


if __name__ == "__main__":
    main()
