"""Streamlit dashboard (FYP proposal §6, §12.2) — animated, PKR-aware build.

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
#  Global styling — one CSS injection drives the whole look: dark
#  base (set in .streamlit/config.toml), Space Grotesk / JetBrains
#  Mono type, gradient accents, and the keyframes used by the
#  animated cards further down.
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --violet:#7c5cff; --pink:#ff5ca8; --gold:#f0b93f; --green:#3ddc97; --red:#ff6b6b;
  --panel:#12151d; --panel2:#171b26; --line:#232838; --dim:#8b93a7; --faint:#565f74;
}
html, body, [class*="css"]{ font-family:'Space Grotesk', sans-serif; }
.mono{ font-family:'JetBrains Mono', monospace; }

/* Hero */
.hero{
  padding:34px 36px; border-radius:20px; margin-bottom:28px;
  background:
    radial-gradient(ellipse 600px 260px at 10% 0%, rgba(124,92,255,.28), transparent 60%),
    radial-gradient(ellipse 500px 260px at 100% 20%, rgba(255,92,168,.18), transparent 55%),
    var(--panel);
  border:1px solid var(--line);
  opacity:0; animation:rise .7s ease forwards;
}
.hero h1{ font-size:32px; font-weight:700; letter-spacing:-.02em; margin:0 0 8px; }
.hero p{ color:var(--dim); font-size:15px; max-width:640px; margin:0; }
.eyebrow{
  font-family:'JetBrains Mono',monospace; font-size:12px; color:var(--gold);
  display:flex; align-items:center; gap:8px; margin-bottom:14px;
}
.live-dot{ width:6px; height:6px; border-radius:50%; background:var(--green); animation:pulse 2s infinite; }
@keyframes pulse{0%,100%{opacity:1;}50%{opacity:.35;}}
@keyframes rise{ from{opacity:0; transform:translateY(12px);} to{opacity:1; transform:none;} }

/* Result tier badge */
.tier-badge{
  text-align:center; padding:24px 10px; border-radius:16px; margin-bottom:16px;
  background:linear-gradient(135deg, rgba(124,92,255,.16), rgba(255,92,168,.08));
  border:1px solid rgba(124,92,255,.32);
  opacity:0; animation:rise .5s ease forwards;
}
.tier-badge .k{ font-family:'JetBrains Mono',monospace; font-size:11.5px; color:var(--dim); }
.tier-badge .name{
  font-size:30px; font-weight:700; margin-top:4px;
  background:linear-gradient(135deg,#fff,var(--dim)); -webkit-background-clip:text; background-clip:text; color:transparent;
}

/* Probability / SHAP bars (CSS-variable driven keyframe fill) */
.bar-row{ margin-bottom:11px; }
.bar-label{ display:flex; justify-content:space-between; font-size:12.5px; color:var(--dim); margin-bottom:5px; font-family:'JetBrains Mono',monospace; }
.bar-track{ height:8px; border-radius:4px; background:var(--panel2); overflow:hidden; }
.bar-fill{ height:100%; border-radius:4px; width:0; background:linear-gradient(90deg, var(--violet), var(--pink));
  animation:grow 1s cubic-bezier(.16,1,.3,1) forwards; animation-delay:.15s; }
@keyframes grow{ to{ width:var(--w); } }

.shap-row{ display:flex; align-items:center; gap:10px; margin-bottom:8px; font-size:12.5px; }
.shap-feat{ width:150px; flex-shrink:0; font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--dim);
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.shap-track{ flex:1; height:16px; background:var(--panel2); border-radius:4px; position:relative; overflow:hidden; }
.shap-mid{ position:absolute; left:50%; top:0; bottom:0; width:1px; background:var(--line); }
.shap-fill{ position:absolute; top:0; bottom:0; width:0; animation:growshap .8s ease forwards; animation-delay:.2s; }
.shap-fill.pos{ left:50%; background:rgba(61,220,151,.55); }
.shap-fill.neg{ right:50%; background:rgba(255,107,107,.55); }
@keyframes growshap{ to{ width:var(--w); } }
.shap-amt{ width:60px; text-align:right; font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--faint); }

/* Market cards */
.market-card{
  background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px;
  opacity:0; animation:rise .5s ease forwards;
}
.market-card .name{ font-size:13.5px; font-weight:600; margin-bottom:8px; line-height:1.3; }
.market-card .price{ font-family:'JetBrains Mono',monospace; font-size:16px; color:var(--gold); font-weight:600; }
.market-card .avail{ font-family:'JetBrains Mono',monospace; font-size:10.5px; color:var(--faint); margin-top:6px; }
.market-empty{ border:1px dashed var(--line); border-radius:12px; padding:32px 20px; text-align:center; color:var(--dim); font-size:13.5px; }
.market-empty code{ font-family:'JetBrains Mono',monospace; background:var(--panel2); padding:2px 7px; border-radius:5px; color:var(--gold); font-size:12px; }

.rate-note{ font-family:'JetBrains Mono',monospace; font-size:11.5px; color:var(--faint); display:flex; align-items:center; gap:6px; margin:2px 0 18px; }
.rate-note.approx .live-dot{ background:var(--faint); }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  Hero
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
  <div class="eyebrow"><span class="live-dot"></span> live · ML-powered · PKR pricing</div>
  <h1>📱 Mobile Price Prediction</h1>
  <p>Predict a smartphone's <b>price tier</b> and <b>estimated price</b> (USD &amp; live PKR) from its hardware
  specs, with SHAP explanations and a live Pakistani market comparison.
  Best classifier: <b>{svc.meta['best_classifier']}</b> · Best regressor: <b>{svc.meta['best_regressor']}</b>.</p>
</div>
""", unsafe_allow_html=True)

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
#  Animated count-up price cards (self-contained HTML/JS component)
# ══════════════════════════════════════════════════════════════════
def price_cards_html(usd: float, pkr: float, live: bool, rate: float) -> str:
    dot_color = "#3ddc97" if live else "#565f74"
    label = "live rate" if live else "approx., live rate unavailable"
    return f"""
    <div style="font-family:'Space Grotesk',sans-serif; background:transparent;">
      <div style="display:flex; gap:10px;">
        <div style="flex:1; background:#171b26; border:1px solid #232838; border-radius:10px; padding:14px;">
          <div style="font-family:'JetBrains Mono',monospace; font-size:11px; color:#565f74;">USD</div>
          <div id="usd" style="font-size:22px; font-weight:700; margin-top:4px; font-family:'JetBrains Mono',monospace; color:#f0b93f;">$0</div>
        </div>
        <div style="flex:1; background:#171b26; border:1px solid #232838; border-radius:10px; padding:14px;">
          <div style="font-family:'JetBrains Mono',monospace; font-size:11px; color:#565f74;">PKR</div>
          <div id="pkr" style="font-size:22px; font-weight:700; margin-top:4px; font-family:'JetBrains Mono',monospace; color:#f0b93f;">Rs 0</div>
        </div>
      </div>
      <div style="font-family:'JetBrains Mono',monospace; font-size:11px; color:#565f74; margin-top:10px; display:flex; align-items:center; gap:6px;">
        <span style="width:5px;height:5px;border-radius:50%;background:{dot_color};display:inline-block;"></span>
        1 USD &asymp; {rate:.2f} PKR ({label})
      </div>
    </div>
    <script>
      function countUp(id, target, prefix) {{
        const el = document.getElementById(id);
        const start = performance.now(), dur = 900;
        function frame(now) {{
          const p = Math.min(1, (now - start) / dur);
          const eased = 1 - Math.pow(1 - p, 3);
          const val = Math.round(target * eased);
          el.textContent = prefix + val.toLocaleString();
          if (p < 1) requestAnimationFrame(frame);
        }}
        requestAnimationFrame(frame);
      }}
      countUp("usd", {usd}, "$");
      countUp("pkr", {pkr}, "Rs ");
    </script>
    """


def bar_html(rows: list[tuple[str, float]]) -> str:
    out = []
    for label, pct in rows:
        out.append(f"""
        <div class="bar-row">
          <div class="bar-label"><span>{label}</span><span>{pct*100:.1f}%</span></div>
          <div class="bar-track"><div class="bar-fill" style="--w:{pct*100}%"></div></div>
        </div>""")
    return "\n".join(out)


def shap_html(contribs: list[dict]) -> str:
    if not contribs:
        return "<p style='color:#565f74; font-size:12.5px;'>Not available for this model.</p>"
    max_abs = max(abs(c["contribution_usd"]) for c in contribs) or 1
    rows = []
    for c in contribs:
        pct = abs(c["contribution_usd"]) / max_abs * 48
        cls = "pos" if c["contribution_usd"] >= 0 else "neg"
        sign = "+" if c["contribution_usd"] >= 0 else ""
        rows.append(f"""
        <div class="shap-row">
          <span class="shap-feat" title="{c['feature']}">{c['feature']}</span>
          <div class="shap-track"><div class="shap-mid"></div><div class="shap-fill {cls}" style="--w:{pct}%"></div></div>
          <span class="shap-amt">{sign}${c['contribution_usd']}</span>
        </div>""")
    return "\n".join(rows)


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

        st.markdown(
            f'<div class="tier-badge"><div class="k">predicted tier</div>'
            f'<div class="name">{tier}</div></div>',
            unsafe_allow_html=True,
        )

        components.html(
            price_cards_html(res["estimated_price_usd"], pkr_price, fx["live"], fx["rate"]),
            height=125,
        )
        st.caption("Price is trained on a demonstration proxy target — see README. "
                   "Compare it with real listings in the Pakistani Market panel →")

        st.markdown("#### Class probabilities")
        proba_rows = list(res["class_probabilities"].items())
        st.markdown(bar_html(proba_rows), unsafe_allow_html=True)

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
        st.markdown(
            '<div class="market-empty">No market data cached yet.<br><br>'
            'Run <code>python -m src.data.scrape_priceoye</code> once '
            '(from a machine with internet access) to populate live '
            'PriceOye.pk listings here.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption(f"{len(data['items'])} listings cached")
        cards = ""
        for i, item in enumerate(data["items"]):
            price = f"Rs {item['price_pkr']:,}" if item.get("price_pkr") else "—"
            cards += (
                f'<div class="market-card" style="animation-delay:{min(i*0.05,0.6)}s; margin-bottom:10px;">'
                f'<div class="name">{item.get("model","Unknown model")}</div>'
                f'<div class="price">{price}</div>'
                f'<div class="avail">{item.get("availability","")}</div></div>'
            )
        st.markdown(cards, unsafe_allow_html=True)
