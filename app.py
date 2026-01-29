import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="EWO Caronte Lab", layout="wide")
st.title("EWO_CARONTE_WLF — Signal Lab")

# -----------------------------
# Helpers
# -----------------------------
def load_jsonl_bytes(b: bytes):
    rows = []
    bad = 0
    for line in b.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            bad += 1
    return rows, bad

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    else:
        df["ts"] = pd.NaT

    for c in ["bar","o","h","l","c","ewo","sig","bandAbs","atr"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "insideBand" in df.columns:
        df["insideBand"] = df["insideBand"].astype("boolean")

    df["date"] = df["ts"].dt.date
    return df

def safe_unique(df, col):
    if col not in df.columns:
        return []
    return sorted([x for x in df[col].dropna().unique().tolist()])

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

# -----------------------------
# Upload
# -----------------------------
st.sidebar.header("Load JSONL")
uploads = st.sidebar.file_uploader(
    "Upload one or more .jsonl files",
    type=["jsonl"],
    accept_multiple_files=True
)

df = pd.DataFrame()
bad_total = 0

if uploads:
    all_rows = []
    for u in uploads:
        rows, bad = load_jsonl_bytes(u.getvalue())
        bad_total += bad
        all_rows.extend(rows)
    df = pd.DataFrame(all_rows)

df = normalize_df(df)

# -----------------------------
# Status
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Files", len(uploads) if uploads else 0)
c2.metric("Rows", int(len(df)))
c3.metric("Bad lines", int(bad_total))
c4.metric("Unique instruments", int(df["instrument"].nunique()) if "instrument" in df.columns and not df.empty else 0)

if df.empty:
    st.info("Upload JSONL files to begin.")
    st.stop()

# -----------------------------
# Filters
# -----------------------------
st.sidebar.header("Filters")

insts = safe_unique(df, "instrument")
tfs   = safe_unique(df, "tf")
evs   = safe_unique(df, "event")
dirs  = safe_unique(df, "dir")

sel_inst = st.sidebar.multiselect("Instrument", insts, default=insts[:1] if insts else [])
sel_tf   = st.sidebar.multiselect("TF", tfs, default=tfs[:1] if tfs else [])
sel_ev   = st.sidebar.multiselect("Event", evs, default=evs)
sel_dir  = st.sidebar.multiselect("Dir", dirs, default=dirs)

min_ts = df["ts"].min()
max_ts = df["ts"].max()
date_range = st.sidebar.date_input(
    "Date range",
    value=(min_ts.date(), max_ts.date()) if pd.notna(min_ts) and pd.notna(max_ts) else None
)

f = df.copy()
if sel_inst and "instrument" in f.columns: f = f[f["instrument"].isin(sel_inst)]
if sel_tf and "tf" in f.columns:           f = f[f["tf"].isin(sel_tf)]
if sel_ev and "event" in f.columns:        f = f[f["event"].isin(sel_ev)]
if sel_dir and "dir" in f.columns:         f = f[f["dir"].isin(sel_dir)]

if isinstance(date_range, (list, tuple)) and len(date_range) == 2 and pd.notna(f["ts"]).any():
    d0, d1 = date_range
    f = f[(f["ts"].dt.date >= d0) & (f["ts"].dt.date <= d1)]

# -----------------------------
# Views
# -----------------------------
st.subheader("Filtered summary")
a, b, c = st.columns(3)
a.metric("Filtered rows", int(len(f)))
b.metric("Days", int(f["date"].nunique()) if "date" in f.columns else 0)
c.metric("Event types", int(f["event"].nunique()) if "event" in f.columns else 0)

if "date" in f.columns and "event" in f.columns:
    counts = f.groupby(["date","event"]).size().reset_index(name="count")
    pivot = counts.pivot(index="date", columns="event", values="count").fillna(0)
    st.write("Events per day")
    st.line_chart(pivot)

st.subheader("Table preview")
show_cols = [c for c in ["ts","instrument","tf","event","dir","bar","o","h","l","c","ewo","sig","bandAbs","insideBand","atr","ver"] if c in f.columns]
st.dataframe(f[show_cols].sort_values("ts").tail(500), use_container_width=True)

st.download_button(
    "Download filtered CSV",
    data=df_to_csv_bytes(f),
    file_name="caronte_filtered.csv",
    mime="text/csv",
)
