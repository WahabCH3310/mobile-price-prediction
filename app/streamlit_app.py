"""Streamlit dashboard (FYP proposal §6, §12.2).

Interactive smartphone price predictor with per-prediction SHAP explanations.

Run:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from src.models.predict import EXAMPLE_PHONE, get_service  # noqa: E402

st.set_page_config(page_title="Mobile Price Predictor", page_icon="📱", layout="wide")

TIER_COLORS = {"Budget": "#6c757d", "Mid": "#0d6efd",
               "Premium": "#6f42c1", "Flagship": "#d63384"}


@st.cache_resource
def load_service():
    return get_service()


svc = load_service()

st.title("📱 Mobile Price Prediction")
st.caption(
    "Predict a smartphone's **price tier** and **estimated price** from its "
    "hardware specs — with SHAP explanations. "
    f"Best classifier: **{svc.meta['best_classifier']}** · "
    f"Best regressor: **{svc.meta['best_regressor']}**."
)

# ── Sidebar: specification inputs ─────────────────────────────────
st.sidebar.header("Phone specifications")


def num(label, key, lo, hi, step=1):
    return st.sidebar.number_input(
        label, min_value=lo, max_value=hi, value=EXAMPLE_PHONE[key], step=step, key=key
    )


with st.sidebar:
    st.subheader("Performance")
    ram = num("RAM (MB)", "ram", 256, 16384, 128)
    int_memory = num("Storage (GB)", "int_memory", 2, 1024, 2)
    n_cores = num("CPU cores", "n_cores", 1, 16, 1)
    clock_speed = st.number_input("Clock speed (GHz)", 0.5, 3.5,
                                  float(EXAMPLE_PHONE["clock_speed"]), 0.1)

    st.subheader("Battery & body")
    battery_power = num("Battery (mAh)", "battery_power", 500, 7000, 50)
    mobile_wt = num("Weight (g)", "mobile_wt", 80, 350, 1)
    m_dep = st.number_input("Depth (cm)", 0.1, 1.5, float(EXAMPLE_PHONE["m_dep"]), 0.1)

    st.subheader("Display")
    px_height = num("Pixel height", "px_height", 0, 3000, 10)
    px_width = num("Pixel width", "px_width", 300, 3000, 10)
    sc_h = st.number_input("Screen height (cm)", 5.0, 25.0,
                           float(EXAMPLE_PHONE["sc_h"]), 0.5)
    sc_w = st.number_input("Screen width (cm)", 0.0, 20.0,
                           float(EXAMPLE_PHONE["sc_w"]), 0.5)

    st.subheader("Cameras")
    pc = num("Primary camera (MP)", "pc", 0, 108, 1)
    fc = num("Front camera (MP)", "fc", 0, 64, 1)

    st.subheader("Connectivity")
    c1, c2 = st.columns(2)
    four_g = int(c1.checkbox("4G", bool(EXAMPLE_PHONE["four_g"])))
    three_g = int(c2.checkbox("3G", bool(EXAMPLE_PHONE["three_g"])))
    wifi = int(c1.checkbox("WiFi", bool(EXAMPLE_PHONE["wifi"])))
    blue = int(c2.checkbox("Bluetooth", bool(EXAMPLE_PHONE["blue"])))
    dual_sim = int(c1.checkbox("Dual SIM", bool(EXAMPLE_PHONE["dual_sim"])))
    touch_screen = int(c2.checkbox("Touch", bool(EXAMPLE_PHONE["touch_screen"])))
    talk_time = num("Talk time (hrs)", "talk_time", 2, 30, 1)

specs = dict(
    battery_power=battery_power, blue=blue, clock_speed=clock_speed,
    dual_sim=dual_sim, fc=fc, four_g=four_g, int_memory=int_memory, m_dep=m_dep,
    mobile_wt=mobile_wt, n_cores=n_cores, pc=pc, px_height=px_height,
    px_width=px_width, ram=ram, sc_h=sc_h, sc_w=sc_w, talk_time=talk_time,
    three_g=three_g, touch_screen=touch_screen, wifi=wifi,
)

# ── Main panel: results ───────────────────────────────────────────
if st.sidebar.button("Predict price", type="primary", use_container_width=True):
    res = svc.predict(specs, explain=True)
    tier = res["price_tier"]

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(
            f"<div style='padding:1.2rem;border-radius:12px;"
            f"background:{TIER_COLORS.get(tier,'#333')};color:white;text-align:center'>"
            f"<div style='font-size:0.9rem;opacity:0.85'>Predicted tier</div>"
            f"<div style='font-size:2rem;font-weight:700'>{tier}</div></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.metric("Estimated price (USD)", f"${res['estimated_price_usd']:,.0f}")
        st.caption("Regression target is a demonstration proxy — see README.")

    st.subheader("Class probabilities")
    proba = pd.DataFrame(
        {"tier": list(res["class_probabilities"]),
         "probability": list(res["class_probabilities"].values())}
    )
    fig = px.bar(proba, x="tier", y="probability", color="tier",
                 color_discrete_map=TIER_COLORS, range_y=[0, 1])
    fig.update_layout(showlegend=False, height=320)
    st.plotly_chart(fig, use_container_width=True)

    if res.get("top_contributions"):
        st.subheader("Why this price? (SHAP contributions)")
        contrib = pd.DataFrame(res["top_contributions"])
        contrib["direction"] = contrib["contribution_usd"].apply(
            lambda v: "increases" if v >= 0 else "decreases")
        fig2 = px.bar(contrib.sort_values("contribution_usd"),
                      x="contribution_usd", y="feature", orientation="h",
                      color="direction",
                      color_discrete_map={"increases": "#2a9d8f",
                                          "decreases": "#e76f51"})
        fig2.update_layout(height=340, xaxis_title="Contribution to price (USD)")
        st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("👈 Set the specifications in the sidebar and click **Predict price**.")
    fig_path = Path(__file__).resolve().parents[1] / "reports/figures/09_shap_importance.png"
    if fig_path.exists():
        st.subheader("Global feature importance (SHAP)")
        st.image(str(fig_path), use_container_width=True)
