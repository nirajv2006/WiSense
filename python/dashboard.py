"""WiSense dashboard.

    streamlit run dashboard.py

Reads what ingest.py writes (data/wisense.db + data/live_window.npz) and the
latest evaluation report (reports/latest/). It never touches the serial port,
so it can be opened, closed and refreshed without disturbing data collection.
"""
import json
import time
from datetime import datetime

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from wisense import config, storage
from wisense.features import FEATURE_GROUPS, FEATURE_NAMES

st.set_page_config(page_title="WiSense", page_icon="📶", layout="wide")

LABEL_COLOR = {"low": "#2E9E5B", "medium": "#E0A100", "high": "#D9534F"}
LEVEL_COLOR = {"ECO": "#2E9E5B", "STANDARD": "#3A7BD5", "FULL": "#E07B00"}
LABEL_Y = {c: i for i, c in enumerate(config.CLASSES)}
STALE_AFTER_S = 10

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("📶 WiSense")
st.sidebar.caption("Wi-Fi CSI occupancy estimation for energy-aware building management")
refresh_s = st.sidebar.slider("Live refresh (s)", 1, 10, 2)
history_label = st.sidebar.selectbox("History range", ["Last 15 min", "Last hour", "Last 6 hours",
                                                       "Last 24 hours", "Everything"], index=1)
HISTORY_S = {"Last 15 min": 900, "Last hour": 3600, "Last 6 hours": 6 * 3600,
             "Last 24 hours": 86400, "Everything": None}[history_label]
st.sidebar.divider()
st.sidebar.markdown("**Occupancy bands**")
for c in config.CLASSES:
    st.sidebar.markdown(f"- **{c.capitalize()}**: {config.CLASS_DEFINITIONS[c]}")
st.sidebar.markdown("**Power levels**")
for c in config.CLASSES:
    lv = config.POWER_LEVELS[c]
    st.sidebar.markdown(f"- **{lv['level']}** ({lv['share']:.0%}) ← {c}")
st.sidebar.caption("WiSense only recommends a level. It does not switch any appliance.")


# ── Helpers ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_model_meta():
    try:
        return joblib.load(config.MODEL_PATH)["meta"]
    except Exception:
        return None


@st.cache_data(ttl=30)
def load_report():
    p = config.REPORTS_DIR / "latest" / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data(ttl=60)
def load_features():
    return pd.read_csv(config.FEATURES_CSV) if config.FEATURES_CSV.exists() else None


def history(seconds):
    since = 0 if seconds is None else time.time() - seconds
    df = storage.read_history(since)
    if not df.empty:
        df["time"] = pd.to_datetime(df.ts, unit="s", utc=True).dt.tz_convert(
            datetime.now().astimezone().tzinfo).dt.tz_localize(None)
    return df


def timeline_figure(df, height=2.6):
    """Occupancy band (coloured steps) + recommended power level over time."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, height), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    has_pred = "label" in df and df.label.notna().any()
    if has_pred:
        d = df[df.label.notna()]
        y = d.label.map(LABEL_Y)
        ax1.step(d.time, y, where="post", color="#555", lw=1, alpha=0.6)
        ax1.scatter(d.time, y, c=d.label.map(LABEL_COLOR), s=10, zorder=3)
        ax1.set_yticks(range(len(config.CLASSES)), [c.capitalize() for c in config.CLASSES])
        ax1.set_ylim(-0.5, len(config.CLASSES) - 0.5)
        ax1.set_ylabel("Occupancy")
        lv = d.power_level.map({l: i for i, l in enumerate(config.POWER_ORDER)})
        ax2.fill_between(d.time, lv, step="post", alpha=0.35, color="#3A7BD5")
        ax2.step(d.time, lv, where="post", color="#3A7BD5")
        ax2.set_yticks(range(len(config.POWER_ORDER)), config.POWER_ORDER, fontsize=8)
        ax2.set_ylim(-0.3, len(config.POWER_ORDER) - 0.7)
        ax2.set_ylabel("Power")
    else:
        ax1.plot(df.time, df.motion, color="#3A7BD5")
        ax1.set_ylabel("Motion index")
        ax2.axis("off")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def csi_heatmap(amp):
    fig, ax = plt.subplots(figsize=(7, 3))
    t = np.arange(amp.shape[0]) / config.EXPECTED_RATE_HZ
    im = ax.imshow(amp.T, aspect="auto", origin="lower", cmap="viridis",
                   extent=[t[0], t[-1], 0, amp.shape[1]],
                   vmin=np.percentile(amp, 2), vmax=np.percentile(amp, 98))
    ax.set_xlabel("Time in window (s)")
    ax.set_ylabel("Subcarrier")
    ax.set_title("CSI amplitude - last window", fontsize=10)
    fig.colorbar(im, ax=ax, label="|H|")
    fig.tight_layout()
    return fig


# ── Header ───────────────────────────────────────────────────────────────────
st.title("WiSense - Occupancy & Power Recommendation")
meta = load_model_meta()
if meta and meta.get("synthetic"):
    st.warning("The loaded model was trained on **synthetic** data. Use it only to test the software; "
               "collect real recordings and re-train before presenting results.", icon="⚠️")

tab_live, tab_hist, tab_model, tab_data = st.tabs(["Live", "History", "Model & evaluation", "Dataset"])

# ── Live ─────────────────────────────────────────────────────────────────────
with tab_live:
    @st.fragment(run_every=refresh_s)
    def live():
        df = history(600)
        snap = storage.read_snapshot()
        if df.empty:
            st.info("No data yet. Start the inference service in another terminal:\n\n"
                    "`python ingest.py --port /dev/cu.usbserial-0001`  (receiver)\n\n"
                    "`python ingest.py --replay \"data/raw/<session>/*.csv\" --loop`  (demo from a recording)")
            return
        last = df.iloc[-1]
        age = time.time() - last.ts
        if age > STALE_AFTER_S:
            st.error(f"No new data for {age:.0f} s. Is ingest.py running and the transmitter powered?")

        has_pred = pd.notna(last.get("label"))
        c1, c2, c3, c4 = st.columns(4)
        if has_pred:
            c1.metric("Occupancy", str(last.label).capitalize(), f"{last.confidence:.0%} confidence",
                      delta_color="off", help=config.CLASS_DEFINITIONS[last.label])
            c2.metric("Recommended power", last.power_level, f"{last.power_share:.0%} of full",
                      delta_color="off")
        else:
            c1.metric("Occupancy", "-", "no model loaded", delta_color="off")
            c2.metric("Recommended power", "-")
        c3.metric("Motion index", f"{last.motion:.4f}",
                  help="Mean temporal std of normalised CSI amplitude (sc_std_mean). Model-independent.")
        c4.metric("Link", f"{last.pkt_rate:.0f} pkt/s", f"RSSI {last.rssi:.0f} dBm", delta_color="off")

        if has_pred:
            info = next(v for v in config.POWER_LEVELS.values() if v["level"] == last.power_level)
            box = {"ECO": st.success, "STANDARD": st.info, "FULL": st.warning}[last.power_level]
            box(f"**{last.power_level}** - {info['actions']}\n\n*Why:* {last.reason}")

        left, right = st.columns([3, 2])
        with left:
            if snap is not None:
                fig = csi_heatmap(snap[0])
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Mean amplitude across subcarriers (last window)")
                st.line_chart(pd.DataFrame({"mean |H|": snap[1]}), height=140)
        with right:
            if has_pred:
                st.markdown("**Class probabilities** (smoothed)")
                st.bar_chart(pd.DataFrame({"probability": [last.p_low, last.p_medium, last.p_high]},
                                          index=config.CLASSES), height=220)
            st.markdown("**Motion index - last 10 min**")
            st.line_chart(df.set_index("time")[["motion"]], height=160)

        st.markdown("**Last 10 minutes**")
        fig = timeline_figure(df)
        st.pyplot(fig)
        plt.close(fig)

    live()

# ── History ──────────────────────────────────────────────────────────────────
with tab_hist:
    @st.fragment(run_every=15)
    def hist():
        df = history(HISTORY_S)
        if df.empty:
            st.info("No history in this range yet.")
            return
        pred = df[df.label.notna()] if "label" in df else df.iloc[0:0]
        span = (df.ts.iloc[-1] - df.ts.iloc[0]) / 60
        st.caption(f"{len(df)} decisions over {span:.0f} min ({history_label.lower()}).")
        fig = timeline_figure(df, height=3.6)
        st.pyplot(fig)
        plt.close(fig)
        if pred.empty:
            return

        a, b = st.columns(2)
        with a:
            st.markdown("**Time in each occupancy band**")
            occ = (pred.label.value_counts().reindex(config.CLASSES).fillna(0) * config.STEP_S / 60)
            st.bar_chart(occ.rename("minutes"), height=220)
        with b:
            st.markdown("**Time at each recommended power level**")
            lvl = (pred.power_level.value_counts().reindex(config.POWER_ORDER).fillna(0) * config.STEP_S / 60)
            st.bar_chart(lvl.rename("minutes"), height=220)

        full_min = lvl.get("FULL", 0)
        total = lvl.sum()
        nominal = (pred.power_share.mean() if total else 1.0)
        m1, m2, m3 = st.columns(3)
        m1.metric("Time below FULL", f"{(total - full_min) / max(total, 1e-9):.0%}")
        m2.metric("Mean recommended level", f"{nominal:.0%} of full")
        m3.metric("Level changes", int((pred.power_level != pred.power_level.shift()).sum() - 1))
        st.caption("These are *recommendation statistics*, not measured energy savings. Real savings depend "
                   "on the connected loads and would need metering to verify.")

        if pred.time.dt.floor("h").nunique() > 1:
            st.markdown("**Occupancy by hour**")
            by_hour = (pred.assign(hour=pred.time.dt.strftime("%H:00"))
                       .groupby(["hour", "label"]).size().unstack(fill_value=0)
                       .reindex(columns=config.CLASSES, fill_value=0))
            st.bar_chart(by_hour, height=240)

        st.download_button("Download history (CSV)", df.drop(columns=["time"]).to_csv(index=False),
                           file_name=f"wisense_history_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv")

    hist()

# ── Model & evaluation ───────────────────────────────────────────────────────
with tab_model:
    rep = load_report()
    if rep is None:
        st.info("No evaluation report yet. Run `python build_dataset.py` then `python train.py`.")
    else:
        m = rep["meta"]
        key = f"{m['model_name']}|{m['primary_scheme']}"
        best = rep["results"][key]
        if m.get("synthetic"):
            st.warning("This report was produced on **synthetic** data and is not a result of WiSense.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Deployed model", m["model_name"])
        c2.metric(f"Macro-F1 (unseen {m['primary_scheme']})", f"{best['f1_macro']:.3f}")
        c3.metric("Accuracy", f"{best['accuracy']:.3f}")
        c4.metric("Training windows", m["n_windows"], f"{len(m['sessions'])} sessions", delta_color="off")
        st.caption(f"Trained {m['trained_at']}. Window {m['window_s']:.0f} s, step {m['step_s']:.0f} s. "
                   "Headline numbers use data from sessions the model never saw during training.")

        st.subheader("Model comparison (macro-F1)")
        comp = pd.DataFrame(rep["comparison_f1"])
        st.dataframe(comp.style.format("{:.3f}").highlight_max(axis=0, color="#cfe8d5"))
        st.caption("'random' mixes overlapping windows of the same recording into train and test, so it is "
                   "optimistic. The gap between 'random' and 'session' shows how much of a score is leakage.")

        rdir = config.REPORTS_DIR / "latest"
        a, b = st.columns(2)
        with a:
            img = rdir / f"confusion_{m['model_name'].lower().replace(' ', '_')}.png"
            if img.exists():
                st.image(str(img), caption=f"{m['model_name']} - normalised confusion matrix")
        with b:
            img = rdir / "confusion_variance_threshold.png"
            if img.exists():
                st.image(str(img), caption="Baseline: variance threshold (current firmware method)")

        a, b = st.columns(2)
        with a:
            st.subheader("Feature importance")
            imp = pd.Series(rep["permutation_importance"]).sort_values(ascending=False).head(12)
            st.bar_chart(imp.rename("drop in macro-F1"), height=300)
        with b:
            st.subheader("Feature-group ablation")
            if rep["ablation_f1"]:
                st.dataframe(pd.Series(rep["ablation_f1"], name="macro-F1").to_frame().style.format("{:.3f}"))
            st.caption("'level' = absolute amplitude and RSSI, which depend on the room. If removing them "
                       "barely hurts, the model relies on movement, which transfers better between rooms.")
        if (rdir / "feature_distributions.png").exists():
            st.image(str(rdir / "feature_distributions.png"))
        if (rdir / "report.md").exists():
            st.download_button("Download full report (Markdown)", (rdir / "report.md").read_text(),
                               file_name="wisense_report.md")

# ── Dataset ──────────────────────────────────────────────────────────────────
with tab_data:
    fdf = load_features()
    if fdf is None:
        st.info("No feature table yet. Record with `collect.py`, then run `python build_dataset.py`.")
    else:
        st.subheader("Windows per session and class")
        st.dataframe(fdf.groupby(["session", "label"]).size().unstack(fill_value=0)
                     .reindex(columns=config.CLASSES, fill_value=0))
        recs = (fdf.groupby("recording")
                .agg(session=("session", "first"), label=("label", "first"), people=("people", "first"),
                     room=("room", "first"), condition=("condition", "first"),
                     windows=("label", "size"), pkt_rate=("pkt_rate", "mean"))
                .sort_values(["session", "label"]))
        st.subheader("Recordings")
        st.dataframe(recs.style.format({"pkt_rate": "{:.1f}"}))

        st.subheader("Feature explorer")
        group = st.selectbox("Feature group", list(FEATURE_GROUPS))
        feat = st.selectbox("Feature", FEATURE_GROUPS[group])
        fig, ax = plt.subplots(figsize=(7, 3))
        sessions = sorted(fdf.session.unique())
        width = 0.8 / max(len(sessions), 1)
        for i, s in enumerate(sessions):
            d = fdf[fdf.session == s]
            pos = np.arange(len(config.CLASSES)) + (i - (len(sessions) - 1) / 2) * width
            ax.boxplot([d.loc[d.label == c, feat] for c in config.CLASSES], positions=pos,
                       widths=width * 0.9, showfliers=False, patch_artist=True,
                       boxprops={"facecolor": plt.cm.tab10(i % 10), "alpha": 0.6})
        ax.set_xticks(range(len(config.CLASSES)), config.CLASSES)
        ax.set_title(f"{feat} by class (one colour per session)", fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Good features separate the classes *consistently in every session*. If the boxes move a "
                   "lot between sessions, the feature is environment-dependent and may cause overfitting.")
