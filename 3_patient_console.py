"""
RA FlareGuard — Interactive Step-by-Step Streamlit UI
======================================================
Clinical standards:
  RAPID3  · Pincus et al., J Rheumatol 2008;35:2136
  MDHAQ   · Pincus & Swearingen, Arthritis Rheum 1999;42:2220
  HAQ-DI  · Fries et al., Arthritis Rheum 1980;23:137
  VAS     · Huskisson, Lancet 1974;2:1127
  Fatigue VAS · Wolfe et al., J Rheumatol 1996;23:1407
  RADAI   · Stucki et al., Arthritis Rheum 1995;38:795
"""

import streamlit as st
import numpy as np
import requests
import joblib
import os
from datetime import datetime, timedelta

st.set_page_config(
    page_title="RA FlareGuard",
    page_icon="🫀",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Fraunces:ital,wght@0,300;0,600;1,300&display=swap');

:root{
  --bg:#070b14; --surface:#0e1420; --card:#141b28;
  --border:#1d2a3f; --border2:#263245;
  --teal:#2dd4bf; --teal2:#0f766e;
  --blue:#60a5fa; --violet:#a78bfa;
  --amber:#fbbf24; --amber-bg:#2a1500;
  --red:#f87171;   --red-bg:#1f0303;
  --green:#4ade80; --green-bg:#031a0a;
  --text:#e2eaf5;  --text2:#7a90b0; --text3:#3d4f68;
  --r:14px; --r2:8px;
}

*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}

html,body,[class*="css"]{
  font-family:'Inter',sans-serif!important;
  background:var(--bg)!important;
  color:var(--text)!important;
  -webkit-font-smoothing:antialiased;
}

#MainMenu,footer,header{visibility:hidden}
.block-container{
  padding:0!important;
  max-width:680px!important;
  margin:0 auto!important;
}

/* ── Progress bar ── */
.prog-wrap{
  position:sticky;top:0;z-index:100;
  background:var(--bg);border-bottom:1px solid var(--border);
  padding:10px 0 8px;
}
.prog-inner{max-width:680px;margin:0 auto;padding:0 24px}
.prog-row{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px}
.prog-label{font-size:0.7rem;font-weight:600;color:var(--teal);text-transform:uppercase;letter-spacing:0.1em}
.prog-step{font-size:0.7rem;color:var(--text3)}
.prog-track{height:3px;background:var(--border);border-radius:99px;overflow:hidden}
.prog-fill{height:100%;border-radius:99px;background:linear-gradient(90deg,var(--teal2),var(--teal));transition:width 0.4s ease}

/* ── Step wrapper ── */
.step-wrap{padding:18px 24px 20px}

/* ── Step header ── */
.step-eyebrow{font-size:0.6rem;font-weight:600;text-transform:uppercase;letter-spacing:0.14em;color:var(--teal);margin-bottom:6px}
.step-title{font-family:'Fraunces',serif;font-size:1.9rem;color:var(--text);line-height:1.1;margin-bottom:6px}
.step-title em{color:var(--teal);font-style:italic}
.step-desc{font-size:0.8rem;color:var(--text2);line-height:1.6;margin-bottom:12px}
.step-cite{display:inline-flex;align-items:center;gap:5px;background:rgba(45,212,191,0.06);border:1px solid rgba(45,212,191,0.15);border-radius:5px;padding:3px 9px;font-size:0.62rem;color:var(--teal);margin-bottom:10px}

/* ── Card ── */
.qcard{background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:14px 18px;margin-bottom:10px}
.qcard-hdr{font-size:0.65rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:var(--text2);margin-bottom:8px;display:flex;align-items:center;gap:6px}
.qcard-hdr::before{content:'';width:3px;height:11px;background:var(--teal);border-radius:2px;display:inline-block}

/* ── Q item ── */
.q-item{padding:7px 0;border-bottom:1px solid var(--border)}
.q-item:last-child{border-bottom:none}
.q-top{display:flex;gap:8px;margin-bottom:6px}
.q-badge{width:20px;height:20px;background:var(--surface);border:1px solid var(--border2);border-radius:5px;display:flex;align-items:center;justify-content:center;font-size:0.58rem;font-weight:700;color:var(--teal);flex-shrink:0;margin-top:1px}
.q-text{font-size:0.8rem;color:var(--text);line-height:1.4}
.q-ref{font-size:0.6rem;color:var(--text3);margin-top:2px}

/* ── VAS slider visual ── */
.vas-grad{height:5px;border-radius:3px;background:linear-gradient(to right,#4ade80,#fbbf24,#f87171);margin:4px 0 2px}
.vas-ends{display:flex;justify-content:space-between;font-size:0.62rem;color:var(--text3);margin-bottom:6px}

/* ── Scale chips ── */
.scale-chips{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0 8px}
.scale-chip{display:inline-flex;align-items:center;gap:4px;padding:3px 9px;border-radius:999px;font-size:0.65rem;font-weight:500}

/* ── Nav buttons (in normal document flow, not fixed — avoids Streamlit's
   scroll-container/transform clipping issues that hide fixed elements) ── */
.nav-bar{
  background:transparent;
  border-top:1px solid var(--border);
  padding:12px 0 4px;
  margin-top:4px;
}
.nav-inner{max-width:680px;margin:0 auto;padding:0 24px;display:flex;gap:10px}

/* Streamlit button overrides */
.stButton>button{
  font-family:'Inter',sans-serif!important;
  font-size:0.84rem!important;font-weight:600!important;
  border-radius:10px!important;padding:0.65rem 1.4rem!important;
  transition:all 0.18s!important;border:none!important;
}
.stButton>button[kind="primary"],
.stButton>button:not([kind]){
  background:linear-gradient(135deg,var(--teal2),#1d4ed8)!important;
  color:#fff!important;
  box-shadow:0 3px 16px rgba(13,148,136,0.3)!important;
}
.stButton>button[kind="primary"]:hover,
.stButton>button:not([kind]):hover{
  transform:translateY(-1px)!important;
  box-shadow:0 5px 22px rgba(13,148,136,0.45)!important;
}
.stButton>button[kind="secondary"]{
  background:var(--card)!important;color:var(--text2)!important;
  border:1px solid var(--border2)!important;
}

/* Widget overrides */
div[data-testid="stSlider"] div[role="slider"]{
  background:var(--teal)!important;
  border:2px solid var(--bg)!important;
  box-shadow:0 0 0 3px rgba(45,212,191,0.2)!important;
}
div[data-testid="stSelectSlider"] div[role="slider"]{
  background:var(--blue)!important;
  border:2px solid var(--bg)!important;
  box-shadow:0 0 0 3px rgba(96,165,250,0.2)!important;
}
.stTextInput input,.stNumberInput input{
  background:var(--card)!important;border:1px solid var(--border2)!important;
  color:var(--text)!important;border-radius:9px!important;
  font-family:'Inter',sans-serif!important;font-size:0.84rem!important;
  padding:10px 14px!important;
}
.stTextInput input:focus,.stNumberInput input:focus{
  border-color:var(--teal)!important;
  box-shadow:0 0 0 3px rgba(45,212,191,0.1)!important;
  outline:none!important;
}
.stSelectbox>div>div{
  background:var(--card)!important;border:1px solid var(--border2)!important;
  color:var(--text)!important;border-radius:9px!important;
  font-family:'Inter',sans-serif!important;font-size:0.84rem!important;
}
label{color:var(--text2)!important;font-size:0.76rem!important;font-family:'Inter',sans-serif!important}
.stMarkdown p{color:var(--text2)!important;font-size:0.8rem!important}
.stNumberInput button{background:var(--card)!important;border-color:var(--border2)!important;color:var(--text2)!important}

/* ── Results ── */
.result-hero{text-align:center;padding:28px 0 20px}
.result-name{font-size:0.68rem;font-weight:600;text-transform:uppercase;letter-spacing:0.12em;color:var(--text3);margin-bottom:6px}
.result-score{font-family:'Fraunces',serif;font-size:5.5rem;line-height:1;margin-bottom:6px}
.result-pill{display:inline-block;font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;padding:5px 16px;border-radius:999px;margin-bottom:10px}
.result-pill.high  {background:var(--red-bg);color:var(--red);border:1px solid #4f0b0b}
.result-pill.medium{background:var(--amber-bg);color:var(--amber);border:1px solid #5c2d00}
.result-pill.low   {background:var(--green-bg);color:var(--green);border:1px solid #0a3d1a}
.result-meta{font-size:0.72rem;color:var(--text3)}

/* Forecast row */
.fc-row{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:20px 0}
.fc-box{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:14px 8px;text-align:center;position:relative;overflow:hidden}
.fc-box.now{border-color:rgba(45,212,191,0.35)}
.fc-box.now::before{content:'NOW';position:absolute;top:0;left:0;right:0;background:var(--teal2);color:#fff;font-size:0.48rem;font-weight:800;letter-spacing:0.2em;padding:3px 0}
.fc-day{font-size:0.58rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:var(--text3);margin-top:4px}
.fc-pct{font-family:'Fraunces',serif;font-size:2.2rem;line-height:1.1;margin:3px 0}
.fc-badge{font-size:0.52rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;padding:2px 7px;border-radius:999px;display:inline-block;margin-bottom:4px}
.fc-badge.high  {background:var(--red-bg);color:var(--red);border:1px solid #4f0b0b}
.fc-badge.medium{background:var(--amber-bg);color:var(--amber);border:1px solid #5c2d00}
.fc-badge.low   {background:var(--green-bg);color:var(--green);border:1px solid #0a3d1a}
.fc-wx{font-size:0.58rem;color:var(--text3);line-height:1.4}

/* R3 bars */
.r3-card{background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:18px 20px;margin:12px 0}
.r3-row{margin-bottom:12px}
.r3-top{display:flex;justify-content:space-between;font-size:0.72rem;margin-bottom:4px}
.r3-lbl{color:var(--text2)}.r3-val{color:var(--text);font-weight:500}
.r3-track{height:5px;background:var(--surface);border-radius:99px;overflow:hidden}
.r3-fill{height:100%;border-radius:99px;transition:width 0.6s ease}
.r3-total{border-top:1px solid var(--border);padding-top:12px;margin-top:4px}
.r3-cats{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-top:12px}
.r3-cat{border-radius:7px;padding:7px 4px;text-align:center}
.r3-catlbl{font-size:0.5rem;color:var(--text3);text-transform:uppercase;letter-spacing:0.07em}
.r3-catval{font-size:0.68rem;font-weight:600;margin-top:2px}

/* Advisory */
.adv{border-radius:var(--r);padding:18px 20px;margin:12px 0}
.adv.high  {background:var(--red-bg);border:1px solid #4f0b0b}
.adv.medium{background:var(--amber-bg);border:1px solid #5c2d00}
.adv.low   {background:var(--green-bg);border:1px solid #0a3d1a}
.adv-hdr{font-size:0.88rem;font-weight:600;margin-bottom:10px}
.adv ul{list-style:none;padding:0}
.adv li{font-size:0.78rem;color:var(--text);padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.05);display:flex;gap:8px;line-height:1.5}
.adv li:last-child{border-bottom:none}
.adv-dot{flex-shrink:0;margin-top:4px;width:5px;height:5px;border-radius:50%}
.adv-dot.high  {background:var(--red)}
.adv-dot.medium{background:var(--amber)}
.adv-dot.low   {background:var(--green)}

/* Info chips row */
.chips{display:flex;gap:7px;flex-wrap:wrap;margin:12px 0}
.chip{background:var(--card);border:1px solid var(--border);border-radius:7px;padding:4px 11px;font-size:0.68rem;color:var(--text2);display:flex;align-items:center;gap:5px}
.chip b{color:var(--text);font-weight:500}

/* Disclaimer */
.disc{margin-top:14px;padding:12px 14px;background:var(--surface);border:1px solid var(--border);border-radius:9px;font-size:0.65rem;color:var(--text3);line-height:1.6}
.disc b{color:var(--text2)}

/* Restart */
.restart-btn{text-align:center;margin-top:20px}

/* ── Tap-to-select pill buttons (replaces drag sliders for scales) ── */
div[data-testid="stRadio"] > div[role="radiogroup"]{
  display:flex; flex-wrap:wrap; gap:8px; margin:6px 0 4px;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label{
  position:relative;
  flex:1 1 58px;
  min-width:58px;
  display:flex; align-items:center; justify-content:center;
  background:var(--surface);
  border:1.5px solid var(--border2);
  border-radius:9px;
  padding:9px 4px;
  cursor:pointer;
  transition:all .15s ease;
  margin:0!important;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover{
  border-color:var(--teal);
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked){
  background:linear-gradient(135deg,var(--teal2),#1d4ed8);
  border-color:var(--teal);
  box-shadow:0 3px 14px rgba(13,148,136,0.35);
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) p{
  color:#fff!important; font-weight:700!important;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child{
  position:absolute; opacity:0; pointer-events:none; width:1px; height:1px; overflow:hidden;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label p{
  color:var(--text2)!important;
  font-size:0.68rem!important;
  font-weight:600!important;
  text-align:center;
  margin:0!important;
  white-space:nowrap;
}
/* Compact numeric NRS pills (0-10 scales) get tighter min-width to fit a row */
.nrs-wrap div[data-testid="stRadio"] div[role="radiogroup"] > label{
  flex:1 1 30px; min-width:30px; padding:8px 2px;
}
.nrs-wrap div[data-testid="stRadio"] div[role="radiogroup"] > label p{
  font-size:0.72rem!important;
}
/* Reduce Streamlit's default per-element vertical gap so sections feel tighter */
div[data-testid="stVerticalBlock"]{ gap:0.35rem!important; }
div[data-testid="element-container"]{ margin-bottom:0!important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("OWM_API_KEY", "")

CITIES = {
    "Colombo":(30,78,1010),"Kandy":(26,72,1012),"Galle":(29,80,1009),
    "Jaffna":(33,65,1008),"Negombo":(30,77,1010),"Trincomalee":(32,68,1007),
    "Anuradhapura":(31,67,1008),"Matara":(28,81,1009),"Kurunegala":(29,73,1010),
    "Badulla":(22,70,1015),"Ratnapura":(28,82,1009),
}

RAPID3_Qs = [
    ("A","Dress yourself, including tying shoelaces and doing buttons?","HAQ item 1"),
    ("B","Get in and out of bed?","HAQ item 2"),
    ("C","Lift a full cup or glass to your mouth?","HAQ item 5"),
    ("D","Walk outdoors on flat ground?","HAQ item 9"),
    ("E","Wash and dry your entire body?","HAQ item 12"),
    ("F","Bend down to pick up clothing from the floor?","HAQ item 14"),
    ("G","Turn regular faucets (taps) on and off?","HAQ item 16"),
    ("H","Get in and out of a car, bus, train, or vehicle?","HAQ item 18"),
    ("I","Walk two miles or about three kilometres?","HAQ item 10"),
    ("J","Participate in recreational activities as you wish?","HAQ item 20"),
]
DIFF = {0:"No difficulty",1:"Some difficulty",2:"Much difficulty",3:"Unable to do"}
DIFF_C = {0:"#4ade80",1:"#facc15",2:"#fb923c",3:"#f87171"}
DIFF_SHORT = {0:"0 · None",1:"1 · Some",2:"2 · Much",3:"3 · Unable"}

TOTAL_STEPS = 7

# ── Tap-to-select pill widgets (replace drag sliders) ───────────────────────────
def diff_pills(key, current):
    """0-3 difficulty scale as tap-to-select pill buttons."""
    opts = [0,1,2,3]
    idx = opts.index(int(current)) if int(current) in opts else 0
    val = st.radio(key, options=opts, format_func=lambda x: DIFF_SHORT[x],
                    index=idx, horizontal=True, key=key, label_visibility="collapsed")
    return val

def nrs_pills(key, current):
    """0-10 numeric rating scale (validated NRS equivalent of a VAS) as tap-to-select pills."""
    opts = list(range(11))
    cur_i = int(round(float(current)))
    cur_i = min(10, max(0, cur_i))
    st.markdown('<div class="nrs-wrap">', unsafe_allow_html=True)
    val = st.radio(key, options=opts, format_func=lambda x: str(x),
                    index=cur_i, horizontal=True, key=key, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    return float(val)

# ── Session state ──────────────────────────────────────────────────────────────
def init():
    defaults = dict(
        step=0,
        name="", city="Colombo", age=45,
        fn_vals={l:0 for l,_,_ in RAPID3_Qs},
        sleep_diff=0, anxiety_val=0,
        fatigue=3.0, pain=3.0, global_s=3.0,
        flares30=2, duration=5, activity=30, heart_rate=75,
        results=None,
    )
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k]=v
init()

# ── Models ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    d=os.path.join(os.path.dirname(__file__),"models")
    try:
        ra   =joblib.load(os.path.join(d,"ra_model.pkl"))["model"]
        sl   =joblib.load(os.path.join(d,"sleep_model.pkl"))["model"]
        hrv  =joblib.load(os.path.join(d,"hrv_model.pkl"))["model"]
        slra =joblib.load(os.path.join(d,"sl_ra_model.pkl"))["model"]
        meta =joblib.load(os.path.join(d,"meta_model.pkl"))
        return ra,sl,hrv,slra,meta,True
    except Exception:
        return None,None,None,None,None,False

ra_m,sl_m,hrv_m,slra_m,meta_m,MODELS_OK=load_models()

# ── Weather ────────────────────────────────────────────────────────────────────
def wx_today(city):
    if not API_KEY:
        t,h,p=CITIES.get(city,(29,75,1010)); return t,h,p,"estimated"
    try:
        r=requests.get(f"http://api.openweathermap.org/data/2.5/weather?q={city},LK&appid={API_KEY}&units=metric",timeout=5).json()
        return r["main"]["temp"],r["main"]["humidity"],r["main"]["pressure"],r["weather"][0]["description"]
    except Exception:
        t,h,p=CITIES.get(city,(29,75,1010)); return t,h,p,"estimated"

def wx_forecast(city):
    if not API_KEY:
        t,h,p,_=wx_today(city)
        return [{"date":(datetime.now()+timedelta(days=i+1)).strftime("%Y-%m-%d"),
                 "temp":round(t+np.random.uniform(-1.5,1.5),1),
                 "humidity":min(100,h+np.random.randint(-5,8)),
                 "pressure":p+np.random.randint(-3,3),"desc":"estimated"} for i in range(3)]
    try:
        r=requests.get(f"http://api.openweathermap.org/data/2.5/forecast?q={city},LK&appid={API_KEY}&units=metric&cnt=24",timeout=5).json()
        daily={}
        for item in r["list"]:
            d=item["dt_txt"].split(" ")[0]
            daily.setdefault(d,{"t":[],"h":[],"p":[],"desc":[]})
            daily[d]["t"].append(item["main"]["temp"])
            daily[d]["h"].append(item["main"]["humidity"])
            daily[d]["p"].append(item["main"]["pressure"])
            daily[d]["desc"].append(item["weather"][0]["description"])
        today=datetime.now().strftime("%Y-%m-%d"); out=[]
        for ds in sorted(daily):
            if ds==today or len(out)>=3: continue
            dv=daily[ds]
            out.append({"date":ds,"temp":round(sum(dv["t"])/len(dv["t"]),1),
                        "humidity":round(sum(dv["h"])/len(dv["h"])),
                        "pressure":round(sum(dv["p"])/len(dv["p"])),
                        "desc":max(set(dv["desc"]),key=dv["desc"].count)})
        if len(out)>=3: return out
        raise Exception("insufficient forecast data")
    except Exception:
        t,h,p,_=wx_today(city)
        return [{"date":(datetime.now()+timedelta(days=i+1)).strftime("%Y-%m-%d"),
                 "temp":round(t+np.random.uniform(-1.5,1.5),1),
                 "humidity":min(100,h+np.random.randint(-5,8)),
                 "pressure":p+np.random.randint(-3,3),"desc":"estimated"} for i in range(3)]

# ── Risk engine ────────────────────────────────────────────────────────────────
def nrm(v,lo,hi): return float(np.clip((v-lo)/(hi-lo),0,1))

def sprob(model,inputs):
    n=model.n_features_in_; arr=np.array(inputs,dtype=float)
    arr=np.pad(arr,(0,max(0,n-len(arr))))[:n]
    return float(model.predict_proba([arr])[0][1])

def compute(pain,fn_score,gs,fatigue,flares,dur,sd,anx,act,hr,r3,temp,hum,pres):
    sp=fn_score*2.8; stp=(gs/10)*12
    shp=max(0,8-sd*1.8); sqp=max(1,10-sd*3); ss=1+anx*3; ep=max(1,10-gs)
    ra_in=[pain,sp,stp,fatigue,flares,dur,hum,temp,pres,1 if hum>75 else 0]
    sl_in=[round(r3/5.5,2),pain*7.6,sp,pain,fatigue,dur*12,stp,10-ep]
    sl2=[shp,sqp,ss,act,hr,ep]
    rms=max(10,60-ss*4-hr*0.2+sqp*2); sdn=rms*1.3; lhf=1+ss*0.3-sqp*0.1
    hrv=[rms,rms,sdn,rms,rms*0.8,rms/max(sdn,1),hr,
         max(0,50-ss*3),max(0,30-ss*2),rms*0.7,sdn*1.2,ss*0.5,(ss-5)*0.2,lhf]
    cs=(nrm(fn_score,0,10)*0.25+nrm(pain,0,10)*0.25+nrm(gs,0,10)*0.20+
        nrm(flares,0,30)*0.15+nrm(fatigue,0,10)*0.10+(1-nrm(shp,0,12))*0.05)
    if MODELS_OK:
        p1=np.clip(sprob(ra_m,ra_in),0,1); p2=np.clip(sprob(slra_m,sl_in),0,1)
        p3=np.clip(sprob(sl_m,sl2),0,1);   p4=np.clip(sprob(hrv_m,hrv),0,1)
        ms=float(meta_m.predict_proba([[p1,p3,p4,p2]])[0][1])
    else:
        ms=cs*0.8
    hr2=nrm(hum,40,100)*0.15; tr=nrm(max(0,32-temp),0,20)*0.05
    risk=cs*0.65+ms*0.15+hr2*0.15+tr*0.05
    if hum>85: risk=min(1,risk+0.06)
    elif hum>80: risk=min(1,risk+0.03)
    elif hum>75: risk=min(1,risk+0.01)
    if r3>12: risk=min(1,risk+0.08)
    elif r3>6: risk=min(1,risk+0.04)
    if pain>=7 and sd>=2: risk=min(1,risk+0.05)
    if r3<=3 and flares<=2: risk=min(risk,0.25)
    return float(np.clip(risk,0,1))

def trend(pain,fat,sd,anx,fl,day):
    d=max(0,(2-sd)*0.04*day); e=anx*0.05*min(day-1,1); p=fl*0.005*day
    return (float(np.clip(pain+p-e*0.3,0,10)),float(np.clip(fat+d,0,10)),
            float(np.clip(anx-e,0,3)),float(np.clip(sd+d*0.5,0,3)))

def var(n): return {"red":"#f87171","amber":"#fbbf24","green":"#4ade80"}[n]
def rlv(r): return "high" if r>0.60 else "medium" if r>0.30 else "low"
def rcl(r): return {"high":var("red"),"medium":var("amber"),"low":var("green")}[rlv(r)]
def rlb(r): return {"high":"High Risk","medium":"Moderate Risk","low":"Low Risk"}[rlv(r)]
def rem(r): return {"high":"🔴","medium":"🟡","low":"🟢"}[rlv(r)]

def r3cat(s):
    if s<=3:  return "Near Remission","#4ade80"
    elif s<=6: return "Low Severity","#60a5fa"
    elif s<=12:return "Moderate Severity","#fbbf24"
    else:      return "High Severity","#f87171"

# ── Progress bar ───────────────────────────────────────────────────────────────
step=st.session_state.step
if step < TOTAL_STEPS:
    pct = int(step / TOTAL_STEPS * 100)
    step_names=["Welcome","Identity","Function","Wellbeing","Pain","Clinical","Results"]
    current_name=step_names[step] if step<len(step_names) else ""
    st.markdown(f"""
    <div class="prog-wrap">
      <div class="prog-inner">
        <div class="prog-row">
          <span class="prog-label">{current_name}</span>
          <span class="prog-step">Step {step+1} of {TOTAL_STEPS}</span>
        </div>
        <div class="prog-track"><div class="prog-fill" style="width:{pct}%"></div></div>
      </div>
    </div>""", unsafe_allow_html=True)

# ── Nav helpers ────────────────────────────────────────────────────────────────
def go(n): st.session_state.step=n; st.rerun()

def nav(back=True, next_label="Continue →", next_fn=None, back_step=None):
    st.markdown('<div class="nav-bar"><div class="nav-inner">', unsafe_allow_html=True)
    if back:
        c1,c2 = st.columns([1,2])
        with c1:
            if st.button("← Back", key=f"back_{step}", type="secondary"):
                go(step-1 if back_step is None else back_step)
        with c2:
            if st.button(next_label, key=f"next_{step}", type="primary"):
                if next_fn: next_fn()
                else: go(step+1)
    else:
        if st.button(next_label, key=f"next_{step}", type="primary"):
            if next_fn: next_fn()
            else: go(step+1)
    st.markdown('</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — WELCOME
# ══════════════════════════════════════════════════════════════════════════════
if step == 0:
    st.markdown("""
    <div class="step-wrap" style="padding-top:8px">
      <div style="text-align:center;padding:8px 0 14px">
        <div style="font-size:0.6rem;font-weight:600;text-transform:uppercase;
             letter-spacing:0.16em;color:var(--teal);margin-bottom:6px">
          RA FlareGuard · Sri Lanka
        </div>
        <div style="font-family:'Fraunces',serif;font-size:1.7rem;color:var(--text);
             line-height:1.1;margin-bottom:8px">
          Know your flare risk<br><em>3 days ahead</em>
        </div>
        <div style="font-size:0.76rem;color:var(--text2);line-height:1.5;
             max-width:420px;margin:0 auto 12px">
          Answer a short clinically validated questionnaire and get a
          personalised 3-day flare risk forecast — including live Sri Lankan
          weather conditions.
        </div>
        <div style="display:flex;gap:6px;justify-content:center;flex-wrap:wrap;margin-bottom:14px">
          <span class="chip">⚕️ RAPID3 / MDHAQ</span>
          <span class="chip">📊 HAQ-DI validated</span>
          <span class="chip">🌤 Live weather</span>
          <span class="chip">🤖 ML models</span>
        </div>
      </div>
      <div class="qcard" style="padding:10px 14px">
        <div class="qcard-hdr" style="margin-bottom:6px">What to expect</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px">
          <div style="background:var(--surface);border:1px solid var(--border);
               border-radius:8px;padding:8px 10px;font-size:0.7rem;color:var(--text2)">
            <b style="color:var(--text)">⏱ 3–5 minutes</b><br>to complete
          </div>
          <div style="background:var(--surface);border:1px solid var(--border);
               border-radius:8px;padding:8px 10px;font-size:0.7rem;color:var(--text2)">
            <b style="color:var(--text)">📋 6 sections</b><br>step by step
          </div>
          <div style="background:var(--surface);border:1px solid var(--border);
               border-radius:8px;padding:8px 10px;font-size:0.7rem;color:var(--text2)">
            <b style="color:var(--text)">🗓 3-day forecast</b><br>with live weather
          </div>
          <div style="background:var(--surface);border:1px solid var(--border);
               border-radius:8px;padding:8px 10px;font-size:0.7rem;color:var(--text2)">
            <b style="color:var(--text)">🔬 Clinical standard</b><br>RAPID3 / HAQ-DI
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    nav(back=False, next_label="Start Assessment →")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — IDENTITY
# ══════════════════════════════════════════════════════════════════════════════
elif step == 1:
    st.markdown("""
    <div class="step-wrap">
      <div class="step-eyebrow">Section 1 of 6</div>
      <div class="step-title">About <em>you</em></div>
      <div class="step-desc">Just a few basics so we can personalise your forecast and fetch live weather for your area.</div>
    </div>""", unsafe_allow_html=True)

    with st.container():
        st.markdown('<div style="padding:0 24px">', unsafe_allow_html=True)
        name = st.text_input("Your full name *", value=st.session_state.name, placeholder="e.g. Rashmi Perera")
        c1,c2 = st.columns(2)
        with c1:
            age = st.number_input("Your age", 18, 90, st.session_state.age)
        with c2:
            city = st.selectbox("Your city", list(CITIES.keys()),
                                index=list(CITIES.keys()).index(st.session_state.city))
        st.markdown('</div>', unsafe_allow_html=True)

    def save1():
        if not name.strip():
            st.error("Please enter your name."); return
        st.session_state.name=name.strip()
        st.session_state.age=age
        st.session_state.city=city
        go(2)

    nav(next_label="Continue →", next_fn=save1)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — RAPID3 FUNCTION (A–J)
# ══════════════════════════════════════════════════════════════════════════════
elif step == 2:
    st.markdown("""
    <div class="step-wrap">
      <div class="step-eyebrow">Section 2 of 6 · RAPID3 / HAQ-DI</div>
      <div class="step-title">What can you <em>do?</em></div>
      <div class="step-desc">
        These 10 questions are from the official RAPID3 questionnaire — the same tool used by
        rheumatologists worldwide to measure physical function in RA patients.
      </div>
      <div class="step-cite">📖 Pincus et al., J Rheumatol 2008;35:2136 · HAQ-DI: Fries et al., Arthritis Rheum 1980</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:0 24px">
    <div class="scale-chips">
      <span class="scale-chip" style="background:#032010;color:#4ade80;border:1px solid #134e23">0 — No difficulty</span>
      <span class="scale-chip" style="background:#1f1500;color:#facc15;border:1px solid #5c3800">1 — Some difficulty</span>
      <span class="scale-chip" style="background:#1f0800;color:#fb923c;border:1px solid #5c2500">2 — Much difficulty</span>
      <span class="scale-chip" style="background:#1f0303;color:#f87171;border:1px solid #5c0b0b">3 — Unable to do</span>
    </div>
    <p style="font-size:0.72rem;color:var(--text3);margin-bottom:16px;font-style:italic">
      Think about the <b style="color:var(--text2)">past 7 days</b> when answering each question.
    </p>
    </div>""", unsafe_allow_html=True)

    fn_vals = dict(st.session_state.fn_vals)
    with st.container():
        st.markdown('<div style="padding:0 24px">', unsafe_allow_html=True)
        for letter, question, ref in RAPID3_Qs:
            cur = fn_vals.get(letter, 0)
            st.markdown(f"""
            <div class="q-item">
              <div class="q-top">
                <div class="q-badge">{letter}</div>
                <div>
                  <div class="q-text">{question}</div>
                  <div class="q-ref">{ref} · MDHAQ 1{letter.lower()}</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)
            v = diff_pills(f"sl_{letter}", cur)
            fn_vals[letter] = v
        st.markdown('</div>', unsafe_allow_html=True)

    fn_raw = sum(fn_vals.values())
    fn_score = round(fn_raw/3, 1)
    st.markdown(f"""
    <div style="padding:0 24px">
    <div style="background:var(--card);border:1px solid var(--border);border-radius:9px;
         padding:12px 16px;margin:12px 0;display:flex;justify-content:space-between;align-items:center">
      <span style="font-size:0.72rem;color:var(--text2)">Function Score (FN)</span>
      <span style="font-size:1.1rem;font-weight:600;color:var(--teal)">{fn_score} / 10</span>
    </div>
    </div>""", unsafe_allow_html=True)

    def save2():
        st.session_state.fn_vals=fn_vals; go(3)
    nav(next_fn=save2)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — MDHAQ SLEEP + ANXIETY  +  FATIGUE VAS
# ══════════════════════════════════════════════════════════════════════════════
elif step == 3:
    st.markdown("""
    <div class="step-wrap">
      <div class="step-eyebrow">Section 3 of 6 · MDHAQ + Fatigue VAS</div>
      <div class="step-title">Sleep, anxiety <em>&amp; fatigue</em></div>
      <div class="step-desc">
        Two questions from the MDHAQ psychological supplement (items K &amp; L), plus the
        validated Fatigue VAS used in rheumatology practice.
      </div>
      <div class="step-cite">📖 MDHAQ: Pincus 2009 · Fatigue VAS: Wolfe et al., J Rheumatol 1996;23:1407</div>
    </div>""", unsafe_allow_html=True)

    with st.container():
        st.markdown('<div style="padding:0 24px">', unsafe_allow_html=True)

        st.markdown("""
        <div class="qcard">
          <div class="qcard-hdr">MDHAQ Supplement — same 0–3 scale</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="q-item">
          <div class="q-top">
            <div class="q-badge">K</div>
            <div>
              <div class="q-text">Get a good night's sleep?</div>
              <div class="q-ref">MDHAQ item 1k · Sleep quality</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
        sd = diff_pills("sd_sl", st.session_state.sleep_diff)

        st.markdown("""
        <div class="q-item" style="margin-top:8px">
          <div class="q-top">
            <div class="q-badge">L</div>
            <div>
              <div class="q-text">Deal with feelings of anxiety or being nervous?</div>
              <div class="q-ref">MDHAQ item 1l · Anxiety</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
        anx = diff_pills("anx_sl", st.session_state.anxiety_val)

        st.markdown("""
        <div class="qcard" style="margin-top:16px">
          <div class="qcard-hdr">Fatigue VAS · Wolfe et al. 1996</div>
          <div style="font-size:0.76rem;color:var(--text2);margin-bottom:8px">
            How much <b style="color:var(--text)">fatigue or tiredness</b> have you had because of your RA <b style="color:var(--text)">over the past week?</b>
          </div>
        </div>""", unsafe_allow_html=True)
        st.markdown('<div class="vas-grad"></div>', unsafe_allow_html=True)
        fat = nrs_pills("fat_sl", st.session_state.fatigue)
        st.markdown('<div class="vas-ends"><span>0 — No fatigue</span><span>10 — Worst possible</span></div>',
                    unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    def save3():
        st.session_state.sleep_diff=sd
        st.session_state.anxiety_val=anx
        st.session_state.fatigue=fat
        go(4)
    nav(next_fn=save3)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — PAIN VAS + GLOBAL
# ══════════════════════════════════════════════════════════════════════════════
elif step == 4:
    st.markdown("""
    <div class="step-wrap">
      <div class="step-eyebrow">Section 4 of 6 · RAPID3 Sections 2 &amp; 3</div>
      <div class="step-title">Pain &amp; <em>overall health</em></div>
      <div class="step-desc">
        Two core RAPID3 measures — Pain VAS and Patient Global Estimate. These are the exact
        questions from the official RAPID3 form used in rheumatology clinics.
      </div>
      <div class="step-cite">📖 Pain VAS: Huskisson, Lancet 1974;2:1127 · RAPID3: Pincus 2008</div>
    </div>""", unsafe_allow_html=True)

    with st.container():
        st.markdown('<div style="padding:0 24px">', unsafe_allow_html=True)

        st.markdown("""
        <div class="qcard">
          <div class="qcard-hdr">RAPID3 Section 2 · Pain VAS</div>
          <div style="font-size:0.76rem;color:var(--text2);margin-bottom:10px">
            <b style="color:var(--text)">Official wording:</b> "How much pain have you had because
            of your condition <b style="color:var(--text)">over the past week?</b>"
          </div>
        </div>""", unsafe_allow_html=True)
        st.markdown('<div class="vas-grad"></div>', unsafe_allow_html=True)
        pain = nrs_pills("pain_sl", st.session_state.pain)
        st.markdown('<div class="vas-ends"><span>0 — No pain at all</span><span>10 — Worst possible pain</span></div>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div class="qcard" style="margin-top:16px">
          <div class="qcard-hdr">RAPID3 Section 3 · Patient Global Estimate</div>
          <div style="font-size:0.76rem;color:var(--text2);margin-bottom:10px">
            <b style="color:var(--text)">Official wording:</b> "Considering all the ways your illness
            affects you at this time — <b style="color:var(--text)">how are you doing overall?</b>"
          </div>
        </div>""", unsafe_allow_html=True)
        st.markdown('<div class="vas-grad" style="background:linear-gradient(to right,#4ade80,#f87171)"></div>',
                    unsafe_allow_html=True)
        gs = nrs_pills("gs_sl", st.session_state.global_s)
        st.markdown('<div class="vas-ends"><span>0 — Very well</span><span>10 — Very poorly</span></div>',
                    unsafe_allow_html=True)

        fn_raw2 = sum(st.session_state.fn_vals.values())
        r3_preview = fn_raw2 + pain + gs
        r3cat_name, r3cat_color = r3cat(r3_preview)
        st.markdown(f"""
        <div style="background:var(--card);border:1px solid var(--border);border-radius:9px;
             padding:12px 16px;margin-top:12px;display:flex;justify-content:space-between;align-items:center">
          <span style="font-size:0.72rem;color:var(--text2)">RAPID3 running total</span>
          <span style="font-size:0.9rem;font-weight:600;color:{r3cat_color}">{r3_preview:.1f}/30 · {r3cat_name}</span>
        </div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    def save4():
        st.session_state.pain=pain; st.session_state.global_s=gs; go(5)
    nav(next_fn=save4)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — CLINICAL HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif step == 5:
    st.markdown("""
    <div class="step-wrap">
      <div class="step-eyebrow">Section 5 of 6 · Clinical History</div>
      <div class="step-title">A bit more <em>history</em></div>
      <div class="step-desc">
        Four quick clinical questions. The flare count is adapted from the RADAI (Rheumatoid Arthritis
        Disease Activity Index). Heart rate is used to estimate your HRV stress signal.
      </div>
      <div class="step-cite">📖 Flare count: RADAI — Stucki et al., Arthritis Rheum 1995;38:795</div>
    </div>""", unsafe_allow_html=True)

    with st.container():
        st.markdown('<div style="padding:0 24px">', unsafe_allow_html=True)

        st.markdown("""<div class="qcard"><div class="qcard-hdr">Flare history — RADAI adapted</div>""",
                    unsafe_allow_html=True)
        st.markdown("""<div style="font-size:0.74rem;color:var(--text2);margin-bottom:8px">
        How many times did your joint pain <b style="color:var(--text)">suddenly get much worse</b>
        in the <b style="color:var(--text)">past 30 days?</b></div>""", unsafe_allow_html=True)
        flares = st.number_input("Flares",0,30,st.session_state.flares30,key="fl_in",label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""<div class="qcard" style="margin-top:10px"><div class="qcard-hdr">RA duration</div>""",
                    unsafe_allow_html=True)
        st.markdown("""<div style="font-size:0.74rem;color:var(--text2);margin-bottom:8px">
        How many years ago were you first diagnosed with RA?</div>""", unsafe_allow_html=True)
        dur = st.number_input("Years",0,50,st.session_state.duration,key="dur_in",label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""<div class="qcard" style="margin-top:10px"><div class="qcard-hdr">Activity today — MDHAQ exercise item</div>""",
                    unsafe_allow_html=True)
        st.markdown("""<div style="font-size:0.74rem;color:var(--text2);margin-bottom:8px">
        How many minutes of <b style="color:var(--text)">walking or light movement</b> did you do today?</div>""",
                    unsafe_allow_html=True)
        act = st.number_input("Minutes",0,180,st.session_state.activity,key="act_in",label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""<div class="qcard" style="margin-top:10px"><div class="qcard-hdr">Heart rate — HRV estimation</div>""",
                    unsafe_allow_html=True)
        st.markdown("""<div style="font-size:0.74rem;color:var(--text2);margin-bottom:8px">
        What is your <b style="color:var(--text)">current pulse rate?</b>
        Check your phone health app, smartwatch, or count for 15 seconds × 4.</div>""", unsafe_allow_html=True)
        hr = st.number_input("bpm",40,180,st.session_state.heart_rate,key="hr_in",label_visibility="collapsed")
        st.markdown("""<div style="font-size:0.65rem;color:var(--text3);margin-top:4px">
        Normal resting pulse = 60 to 100 beats per minute</div></div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    def save5():
        st.session_state.flares30=flares; st.session_state.duration=dur
        st.session_state.activity=act; st.session_state.heart_rate=hr
        fn_raw_f = sum(st.session_state.fn_vals.values())
        fn_sc    = round(fn_raw_f/3,1)
        r3       = fn_raw_f + st.session_state.pain + st.session_state.global_s
        with st.spinner("Fetching live weather & computing forecast…"):
            t,h,p,d = wx_today(st.session_state.city)
            fc       = wx_forecast(st.session_state.city)
        today_risk = compute(st.session_state.pain,fn_sc,st.session_state.global_s,
                             st.session_state.fatigue,flares,dur,
                             st.session_state.sleep_diff,st.session_state.anxiety_val,
                             act,hr,r3,t,h,p)
        risks=[today_risk]
        for i,f in enumerate(fc):
            fp,ff,fs,fsd=trend(st.session_state.pain,st.session_state.fatigue,
                               st.session_state.sleep_diff,st.session_state.anxiety_val,flares,i+1)
            fr3=fn_raw_f+fp+st.session_state.global_s
            risks.append(compute(fp,fn_sc,st.session_state.global_s,ff,flares,dur,
                                 fsd,fs,act,hr,fr3,f["temp"],f["humidity"],f["pressure"]))
        st.session_state.results={
            "fn_raw":fn_raw_f,"fn_score":fn_sc,"r3":r3,
            "pain":st.session_state.pain,"gs":st.session_state.global_s,
            "fatigue":st.session_state.fatigue,
            "risks":risks,"forecast":fc,
            "temp":t,"hum":h,"pres":p,"desc":d,
        }
        go(6)

    nav(next_label="Calculate My Forecast →", next_fn=save5)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif step == 6:
    res = st.session_state.results
    if not res:
        go(0)
    else:
        risks = res["risks"]
        peak  = max(risks)
        plvl  = rlv(peak)
        pcol  = rcl(peak)
        r3    = res["r3"]
        r3name, r3col = r3cat(r3)
        fc    = res["forecast"]

        st.markdown(f"""
        <div class="step-wrap">
          <div class="step-eyebrow">Your 3-Day Forecast · {st.session_state.city}</div>
          <div class="result-hero">
            <div class="result-name">{st.session_state.name} · Age {st.session_state.age}</div>
            <div class="result-score" style="color:{pcol}">{peak:.0%}</div>
            <div class="result-pill {plvl}">{rem(peak)} {rlv(peak).replace('medium','moderate').title()} Risk</div>
            <div class="result-meta">
              {datetime.now().strftime('%d %b %Y')} &nbsp;·&nbsp;
              Peak risk over next 3 days
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="padding:0 24px">
        <div class="chips">
          <span class="chip">📍 <b>{st.session_state.city}</b></span>
          <span class="chip">🌡 <b>{res['temp']:.1f}°C</b> · {res['hum']}% humidity</span>
          <span class="chip">📊 RAPID3 <b style="color:{r3col}">{r3:.1f}/30</b> · {r3name}</span>
          <span class="chip">🤖 <b>{'ML Active' if MODELS_OK else 'Formula Mode'}</b></span>
        </div>
        </div>""", unsafe_allow_html=True)

        day_labels=["Today","Tomorrow","Day +2","Day +3"]
        wxlist=[{"temp":res["temp"],"humidity":res["hum"],"desc":res["desc"]},
                *[{"temp":f["temp"],"humidity":f["humidity"],"desc":f["desc"]} for f in fc]]
        dates=[datetime.now().strftime("%d %b"),*[f["date"] for f in fc]]

        fc_html=""
        for i,(label,risk,w,date) in enumerate(zip(day_labels,risks,wxlist,dates)):
            lvl=rlv(risk); col=rcl(risk); now="now" if i==0 else ""
            emoji=rem(risk); lb=rlv(risk).replace("medium","moderate").title()
            fc_html+=f"""
            <div class="fc-box {now}">
              <div class="fc-day">{label}</div>
              <div style="font-size:0.58rem;color:var(--text3);margin-bottom:4px">{date}</div>
              <div class="fc-pct" style="color:{col}">{risk:.0%}</div>
              <div class="fc-badge {lvl}">{emoji} {lb}</div>
              <div class="fc-wx">{w['temp']:.1f}°C · {w['humidity']}%<br>{w['desc']}</div>
            </div>"""

        st.markdown(f"""
        <div style="padding:0 24px">
        <div class="fc-row">{fc_html}</div>
        </div>""", unsafe_allow_html=True)

        fn_sc=res["fn_score"]; pain=res["pain"]; gs=res["gs"]; fat=res["fatigue"]
        fnp=fn_sc/10*100; pp=pain/10*100; gp=gs/10*100; fp_=fat/10*100; r3p=r3/30*100

        cats=[
            ("≤3","Near Remission","#4ade80","#032010","#134e23",r3<=3),
            ("4–6","Low","#60a5fa","#0d1a3d","#1e3a6b",3<r3<=6),
            ("7–12","Moderate","#fbbf24","#3d1a00","#6b2d00",6<r3<=12),
            (">12","High","#f87171","#2d0404","#6b0505",r3>12),
        ]
        cats_html="".join(f"""<div class="r3-cat" style="background:{bg};
            border:{'2px' if active else '1px'} solid {bdr}">
            <div class="r3-catlbl">{lbl}</div>
            <div class="r3-catval" style="color:{col}">{val}</div>
            </div>""" for val,lbl,col,bg,bdr,active in cats)

        st.markdown(f"""
        <div style="padding:0 24px">
        <div class="r3-card">
          <div class="qcard-hdr">RAPID3 Score Breakdown · Pincus et al. 2008</div>
          <div class="r3-row">
            <div class="r3-top"><span class="r3-lbl">Function Score A–J (HAQ-DI)</span><span class="r3-val">{fn_sc:.1f}/10</span></div>
            <div class="r3-track"><div class="r3-fill" style="width:{fnp:.0f}%;background:#60a5fa"></div></div>
          </div>
          <div class="r3-row">
            <div class="r3-top"><span class="r3-lbl">Pain VAS (Huskisson 1974)</span><span class="r3-val">{pain:.1f}/10</span></div>
            <div class="r3-track"><div class="r3-fill" style="width:{pp:.0f}%;background:#f97316"></div></div>
          </div>
          <div class="r3-row">
            <div class="r3-top"><span class="r3-lbl">Patient Global Estimate</span><span class="r3-val">{gs:.1f}/10</span></div>
            <div class="r3-track"><div class="r3-fill" style="width:{gp:.0f}%;background:#a78bfa"></div></div>
          </div>
          <div class="r3-row" style="opacity:0.7">
            <div class="r3-top"><span class="r3-lbl">Fatigue VAS (Wolfe 1996) — supplementary</span><span class="r3-val">{fat:.1f}/10</span></div>
            <div class="r3-track"><div class="r3-fill" style="width:{fp_:.0f}%;background:#2dd4bf"></div></div>
          </div>
          <div class="r3-total">
            <div class="r3-top">
              <span style="font-weight:600;color:var(--text);font-size:0.75rem">Total RAPID3</span>
              <span style="color:{r3col};font-size:0.88rem;font-weight:600">{r3:.1f}/30 · {r3name}</span>
            </div>
            <div class="r3-track" style="height:7px"><div class="r3-fill" style="width:{r3p:.0f}%;background:{r3col}"></div></div>
          </div>
          <div class="r3-cats">{cats_html}</div>
        </div>
        </div>""", unsafe_allow_html=True)

        if plvl=="high":
            title="🔴  High Flare Risk — Action Recommended"
            items=[
                "Contact your rheumatologist or doctor today — do not wait",
                "Rest as much as possible — avoid heavy lifting",
                "Take your prescribed medicines on time",
                "Apply warm or cold packs to painful joints",
                "Avoid going outside during peak midday heat (11am–3pm)",
            ]
            if res["hum"]>75 or any(f["humidity"]>75 for f in fc):
                items.append(f"High humidity in {st.session_state.city} — stay indoors, use a fan or AC")
        elif plvl=="medium":
            title="🟡  Moderate Risk — Take Precautions"
            items=[
                "Aim for at least 7–8 hours of sleep tonight",
                "Avoid stressful or physically demanding activities",
                "Drink plenty of water — stay well hydrated",
                "Gentle stretching only — no strenuous exercise",
                "Call your doctor if pain suddenly worsens",
            ]
            if any(f["humidity"]>75 for f in fc):
                items.append(f"Elevated humidity expected in {st.session_state.city} — take cool breaks")
        else:
            title="🟢  Low Risk — You Are Doing Well"
            items=[
                "Continue your current routine and medication schedule",
                "Stay hydrated — drink water regularly",
                "Light walking is good for your joints",
                "Attend any planned appointments",
            ]

        adv_html="".join(f'<li><div class="adv-dot {plvl}"></div>{it}</li>' for it in items)
        st.markdown(f"""
        <div style="padding:0 24px">
        <div class="adv {plvl}">
          <div class="adv-hdr">{title}</div>
          <ul>{adv_html}</ul>
        </div>
        <div class="disc">
          ⚕️ &nbsp;<b>Clinical Decision Support Only.</b>
          Based on RAPID3/MDHAQ (Pincus 2008), HAQ-DI (Fries 1980), VAS Pain (Huskisson 1974),
          Fatigue VAS (Wolfe 1996), RADAI (Stucki 1995).
          This tool does <b>not replace</b> evaluation by a qualified rheumatologist.
        </div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div style="padding:0 24px">', unsafe_allow_html=True)
        if st.button("← Start New Assessment", type="secondary"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)