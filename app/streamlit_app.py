"""Streamlit dashboard (FYP proposal §6, §12.2) — professional, PKR-aware build.

Interactive smartphone price predictor with per-prediction SHAP explanations,
a live USD→PKR price conversion, and a Pakistani retail-market comparison
panel. Everything runs inside Streamlit — no separate backend needed.

Run:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

from src.models.predict import EXAMPLE_PHONE, get_service  # noqa: E402
from src.services.currency import get_usd_to_pkr_rate  # noqa: E402
from src.services.market import get_market_data  # noqa: E402

st.set_page_config(page_title="Mobile Price Predictor — Pakistan", page_icon="📱", layout="wide")


@st.cache_resource
def load_service():
    return get_service()


svc = load_service()

# ══════════════════════════════════════════════════════════════════
#  Global styling.
#
#  IMPORTANT: Streamlit runs st.markdown() text through Python-Markdown
#  before allowing raw HTML. Python-Markdown treats a blank line as the
#  end of an HTML block — anything after a blank line inside a <style>
#  tag gets dumped onto the page as literal visible text instead of
#  being applied as CSS. So every string below is built with ZERO blank
#  lines. Do not reformat this with blank lines between rules.
# ══════════════════════════════════════════════════════════════════
_CSS_LINES = [
    "<link rel='preconnect' href='https://fonts.googleapis.com'>",
    "<link href='https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap' rel='stylesheet'>",
    "<style>",
    ":root{--accent:#5b8def;--accent2:#4fd1c5;--gold:#e8b85c;--green:#34d399;--red:#f87171;",
    "--panel:#161920;--panel2:#1c2029;--line:#2a2f3a;--ink:#eef0f3;--dim:#9099a8;--faint:#5b6270;}",
    "html,body,[class*='css']{font-family:'Space Grotesk',sans-serif;}",
    ".mono{font-family:'JetBrains Mono',monospace;}",
    "div[data-testid='stAppViewContainer']{background:radial-gradient(ellipse 900px 500px at 15% -10%,rgba(91,141,239,.10),transparent 60%),radial-gradient(ellipse 700px 460px at 100% 5%,rgba(79,209,197,.07),transparent 55%),#0e1013;}",
    ".card-3d{transition:transform .45s cubic-bezier(.2,.8,.2,1),box-shadow .45s ease;}",
    ".card-3d:hover{transform:perspective(900px) rotateX(2.5deg) rotateY(-2.5deg) translateY(-4px);box-shadow:0 26px 50px -18px rgba(0,0,0,.65),0 0 0 1px rgba(91,141,239,.14);}",
    ".hero{padding:32px 36px;border-radius:18px;margin-bottom:26px;background:linear-gradient(160deg,#171b24 0%,#13161d 100%);border:1px solid var(--line);box-shadow:0 18px 40px -22px rgba(0,0,0,.7);opacity:0;animation:rise .6s ease forwards;}",
    ".hero h1{font-size:30px;font-weight:700;letter-spacing:-.02em;margin:0 0 8px;color:var(--ink);}",
    ".hero p{color:var(--dim);font-size:14.5px;max-width:660px;margin:0;line-height:1.55;}",
    ".eyebrow{font-family:'JetBrains Mono',monospace;font-size:11.5px;color:var(--accent2);display:flex;align-items:center;gap:8px;margin-bottom:14px;letter-spacing:.02em;}",
    ".live-dot{width:6px;height:6px;border-radius:50%;background:var(--green);animation:pulse 2.2s infinite;}",
    "@keyframes pulse{0%,100%{opacity:1;}50%{opacity:.35;}}",
    "@keyframes rise{from{opacity:0;transform:translateY(10px);}to{opacity:1;transform:none;}}",
    ".tier-badge{text-align:center;padding:22px 10px;border-radius:14px;margin-bottom:14px;background:var(--panel);border:1px solid var(--line);opacity:0;animation:rise .45s ease forwards;}",
    ".tier-badge .k{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--dim);text-transform:uppercase;letter-spacing:.06em;}",
    ".tier-badge .name{font-size:27px;font-weight:700;margin-top:6px;color:var(--ink);}",
    ".tier-badge .accent-bar{height:3px;width:40px;margin:12px auto 0;border-radius:2px;background:linear-gradient(90deg,var(--accent),var(--accent2));}",
    ".bar-row{margin-bottom:11px;}",
    ".bar-label{display:flex;justify-content:space-between;font-size:12.5px;color:var(--dim);margin-bottom:5px;font-family:'JetBrains Mono',monospace;}",
    ".bar-track{height:7px;border-radius:4px;background:var(--panel2);overflow:hidden;}",
    ".bar-fill{height:100%;border-radius:4px;width:0;background:linear-gradient(90deg,var(--accent),var(--accent2));animation:grow 1s cubic-bezier(.16,1,.3,1) forwards;animation-delay:.1s;}",
    "@keyframes grow{to{width:var(--w);}}",
    ".shap-row{display:flex;align-items:center;gap:10px;margin-bottom:7px;font-size:12.5px;}",
    ".shap-feat{width:150px;flex-shrink:0;font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--dim);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}",
    ".shap-track{flex:1;height:15px;background:var(--panel2);border-radius:4px;position:relative;overflow:hidden;}",
    ".shap-mid{position:absolute;left:50%;top:0;bottom:0;width:1px;background:var(--line);}",
    ".shap-fill{position:absolute;top:0;bottom:0;width:0;animation:growshap .8s ease forwards;animation-delay:.15s;}",
    ".shap-fill.pos{left:50%;background:rgba(52,211,153,.5);}",
    ".shap-fill.neg{right:50%;background:rgba(248,113,113,.5);}",
    "@keyframes growshap{to{width:var(--w);}}",
    ".shap-amt{width:58px;text-align:right;font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--faint);}",
    ".market-card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:15px;margin-bottom:10px;opacity:0;animation:rise .45s ease forwards;}",
    ".market-card .name{font-size:13px;font-weight:600;margin-bottom:7px;line-height:1.3;color:var(--ink);}",
    ".market-card .price{font-family:'JetBrains Mono',monospace;font-size:15px;color:var(--gold);font-weight:600;}",
    ".market-card .avail{font-family:'JetBrains Mono',monospace;font-size:10.5px;color:var(--faint);margin-top:5px;}",
    ".market-empty{border:1px dashed var(--line);border-radius:12px;padding:30px 18px;text-align:center;color:var(--dim);font-size:13px;}",
    ".market-empty code{font-family:'JetBrains Mono',monospace;background:var(--panel2);padding:2px 6px;border-radius:5px;color:var(--gold);font-size:11.5px;}",
    "div[data-testid='stButton']>button{border-radius:10px;font-weight:600;transition:transform .15s ease,box-shadow .15s ease;box-shadow:0 6px 18px -8px rgba(91,141,239,.55);}",
    "div[data-testid='stButton']>button:hover{transform:translateY(-2px);box-shadow:0 10px 22px -8px rgba(91,141,239,.7);}",
    "div[data-testid='stButton']>button:active{transform:translateY(0px) scale(.98);box-shadow:0 3px 10px -6px rgba(91,141,239,.5);}",
    "</style>",
]
st.markdown("".join(_CSS_LINES), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  Hero
# ══════════════════════════════════════════════════════════════════
_hero = (
    "<div class='hero card-3d'>"
    "<div class='eyebrow'><span class='live-dot'></span> live · ML-powered · PKR pricing</div>"
    "<h1>📱 Mobile Price Prediction</h1>"
    "<p>Predict a smartphone's <b>price tier</b> and <b>estimated price</b> (USD &amp; live PKR) from its "
    "hardware specs, with SHAP explanations and a live Pakistani market comparison. "
    f"Best classifier: <b>{svc.meta['best_classifier']}</b> · Best regressor: <b>{svc.meta['best_regressor']}</b>.</p>"
    "</div>"
)
st.markdown(_hero, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  Sidebar: specification inputs
# ══════════════════════════════════════════════════════════════════
st.sidebar.header("📋 Phone specifications")


def num(label, key, lo, hi, step=1):
    return st.sidebar.number_input(
        label, min_value=lo, max_value=hi, value=EXAMPLE_PHONE[key], step=step, key=key
    )


with st.sidebar:
    st.subheader("⚙️ Performance")
    ram = num("RAM (MB)", "ram", 256, 16384, 128)
    int_memory = num("Storage (GB)", "int_memory", 2, 1024, 2)
    n_cores = num("CPU cores", "n_cores", 1, 16, 1)
    clock_speed = st.number_input("Clock speed (GHz)", 0.5, 3.5,
                                  float(EXAMPLE_PHONE["clock_speed"]), 0.1)

    st.subheader("🔋 Battery & body")
    battery_power = num("Battery (mAh)", "battery_power", 500, 7000, 50)
    mobile_wt = num("Weight (g)", "mobile_wt", 80, 350, 1)
    m_dep = st.number_input("Depth (cm)", 0.1, 1.5, float(EXAMPLE_PHONE["m_dep"]), 0.1)

    st.subheader("🖥️ Display")
    px_height = num("Pixel height", "px_height", 0, 3000, 10)
    px_width = num("Pixel width", "px_width", 300, 3000, 10)
    sc_h = st.number_input("Screen height (cm)", 5.0, 25.0,
                           float(EXAMPLE_PHONE["sc_h"]), 0.5)
    sc_w = st.number_input("Screen width (cm)", 0.0, 20.0,
                           float(EXAMPLE_PHONE["sc_w"]), 0.5)

    st.subheader("📷 Cameras")
    pc = num("Primary camera (MP)", "pc", 0, 108, 1)
    fc = num("Front camera (MP)", "fc", 0, 64, 1)

    st.subheader("📶 Connectivity")
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


# ══════════════════════════════════════════════════════════════════
#  HTML builders — every string below is assembled with NO blank
#  lines, for the same reason described in the CSS section above.
# ══════════════════════════════════════════════════════════════════
def price_cards_html(usd: float, pkr: float, live: bool, rate: float) -> str:
    dot_color = "#34d399" if live else "#5b6270"
    label = "live rate" if live else "approx., live rate unavailable"
    return (
        "<div style=\"font-family:'Space Grotesk',sans-serif;background:transparent;\">"
        "<div style='display:flex;gap:10px;perspective:600px;'>"
        "<div class='pc-3d' style=\"flex:1;background:#1c2029;border:1px solid #2a2f3a;border-radius:10px;padding:14px;transition:transform .35s ease;\">"
        "<div style=\"font-family:'JetBrains Mono',monospace;font-size:11px;color:#5b6270;\">USD</div>"
        "<div id='usd' style=\"font-size:21px;font-weight:700;margin-top:4px;font-family:'JetBrains Mono',monospace;color:#e8b85c;\">$0</div>"
        "</div>"
        "<div class='pc-3d' style=\"flex:1;background:#1c2029;border:1px solid #2a2f3a;border-radius:10px;padding:14px;transition:transform .35s ease;\">"
        "<div style=\"font-family:'JetBrains Mono',monospace;font-size:11px;color:#5b6270;\">PKR</div>"
        "<div id='pkr' style=\"font-size:21px;font-weight:700;margin-top:4px;font-family:'JetBrains Mono',monospace;color:#e8b85c;\">Rs 0</div>"
        "</div>"
        "</div>"
        f"<div style=\"font-family:'JetBrains Mono',monospace;font-size:11px;color:#5b6270;margin-top:10px;display:flex;align-items:center;gap:6px;\">"
        f"<span style='width:5px;height:5px;border-radius:50%;background:{dot_color};display:inline-block;'></span>"
        f"1 USD &asymp; {rate:.2f} PKR ({label})</div>"
        "</div>"
        "<style>.pc-3d:hover{transform:translateY(-3px) scale(1.015);border-color:#5b8def !important;}</style>"
        "<script>"
        "function countUp(id,target,prefix){"
        "const el=document.getElementById(id);const start=performance.now(),dur=900;"
        "function frame(now){const p=Math.min(1,(now-start)/dur);const eased=1-Math.pow(1-p,3);"
        "const val=Math.round(target*eased);el.textContent=prefix+val.toLocaleString();"
        "if(p<1)requestAnimationFrame(frame);}"
        "requestAnimationFrame(frame);}"
        f"countUp('usd',{usd},'$');"
        f"countUp('pkr',{pkr},'Rs ');"
        "</script>"
    )


def bar_html(rows: list[tuple[str, float]]) -> str:
    parts = []
    for label, pct in rows:
        parts.append(
            "<div class='bar-row'>"
            f"<div class='bar-label'><span>{label}</span><span>{pct*100:.1f}%</span></div>"
            f"<div class='bar-track'><div class='bar-fill' style='--w:{pct*100}%'></div></div>"
            "</div>"
        )
    return "".join(parts)


def shap_html(contribs: list[dict]) -> str:
    if not contribs:
        return "<p style='color:#5b6270;font-size:12.5px;'>Not available for this model.</p>"
    max_abs = max(abs(c["contribution_usd"]) for c in contribs) or 1
    parts = []
    for c in contribs:
        pct = abs(c["contribution_usd"]) / max_abs * 48
        cls = "pos" if c["contribution_usd"] >= 0 else "neg"
        sign = "+" if c["contribution_usd"] >= 0 else ""
        parts.append(
            "<div class='shap-row'>"
            f"<span class='shap-feat' title='{c['feature']}'>{c['feature']}</span>"
            f"<div class='shap-track'><div class='shap-mid'></div><div class='shap-fill {cls}' style='--w:{pct}%'></div></div>"
            f"<span class='shap-amt'>{sign}${c['contribution_usd']}</span>"
            "</div>"
        )
    return "".join(parts)


# ══════════════════════════════════════════════════════════════════
#  Main panel: prediction results
# ══════════════════════════════════════════════════════════════════
col_main, col_market = st.columns([1.3, 1])

with col_main:
    if st.sidebar.button("🔮 Predict price", type="primary", use_container_width=True):
        res = svc.predict(specs, explain=True)
        tier = res["price_tier"]
        fx = get_usd_to_pkr_rate()
        pkr_price = round(res["estimated_price_usd"] * fx["rate"])

        badge = (
            "<div class='tier-badge card-3d'>"
            "<div class='k'>predicted tier</div>"
            f"<div class='name'>{tier}</div>"
            "<div class='accent-bar'></div>"
            "</div>"
        )
        st.markdown(badge, unsafe_allow_html=True)

        components.html(
            price_cards_html(res["estimated_price_usd"], pkr_price, fx["live"], fx["rate"]),
            height=125,
        )
        st.caption("Price is trained on a demonstration proxy target — see README. "
                   "Compare it with real listings in the Pakistani Market panel →")

        st.markdown("#### Class probabilities")
        st.markdown(bar_html(list(res["class_probabilities"].items())), unsafe_allow_html=True)

        if res.get("top_contributions"):
            st.markdown("#### Why this price? (SHAP contributions)")
            st.markdown(shap_html(res["top_contributions"]), unsafe_allow_html=True)
    else:
        st.info("👈 Set the specifications in the sidebar and click **Predict price**.")
        fig_path = Path(__file__).resolve().parents[1] / "reports/figures/09_shap_importance.png"
        if fig_path.exists():
            st.markdown("#### Global feature importance (SHAP)")
            st.image(str(fig_path), use_container_width=True)

# ══════════════════════════════════════════════════════════════════
#  Pakistani market comparison panel
# ══════════════════════════════════════════════════════════════════
with col_market:
    st.markdown("#### 🇵🇰 Pakistani market")
    data = get_market_data(limit=12)
    if not data["items"]:
        empty = (
            "<div class='market-empty'>No market data cached yet.<br><br>"
            "Run <code>python -m src.data.scrape_priceoye</code> once "
            "(from a machine with internet access) to populate live "
            "PriceOye.pk listings here.</div>"
        )
        st.markdown(empty, unsafe_allow_html=True)
    else:
        st.caption(f"{len(data['items'])} listings cached")
        cards = []
        for i, item in enumerate(data["items"]):
            price = f"Rs {item['price_pkr']:,}" if item.get("price_pkr") else "—"
            cards.append(
                f"<div class='market-card card-3d' style='animation-delay:{min(i*0.05,0.6)}s;'>"
                f"<div class='name'>{item.get('model','Unknown model')}</div>"
                f"<div class='price'>{price}</div>"
                f"<div class='avail'>{item.get('availability','')}</div>"
                "</div>"
            )
        st.markdown("".join(cards), unsafe_allow_html=True)
