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
import joblib, requests, warnings
warnings.filterwarnings('ignore')

# ── Load all models once ──────────────────────────────────────────────────────
ra_model    = joblib.load("models/ra_model.pkl")["model"]
sleep_model = joblib.load("models/sleep_model.pkl")["model"]
hrv_model   = joblib.load("models/hrv_model.pkl")["model"]
sl_ra_model = joblib.load("models/sl_ra_model.pkl")["model"]
meta_model  = joblib.load("models/meta_model.pkl")

def ask(prompt, lo, hi):
    while True:
        try:
            v = float(input(f"\n  {prompt}\n  👉 Your answer [{lo}-{hi}]: "))
            if lo <= v <= hi:
                return v
            print(f"  ⚠  Please enter a number between {lo} and {hi}")
        except ValueError:
            print("  ⚠  Please type a number.")

def get_name():
    while True:
        name = input("\n  What is your full name? : ").strip()
        if name:
            return name
        print("  ⚠  Name cannot be empty. Please enter your name.")

def choose_location():
    cities = {
        "1":  ("Colombo",      6.9271,  79.8612),
        "2":  ("Kandy",        7.2906,  80.6337),
        "3":  ("Galle",        6.0535,  80.2210),
        "4":  ("Jaffna",       9.6615,  80.0255),
        "5":  ("Negombo",      7.2081,  79.8358),
        "6":  ("Trincomalee",  8.5874,  81.2152),
        "7":  ("Anuradhapura", 8.3114,  80.4037),
        "8":  ("Matara",       5.9549,  80.5550),
        "9":  ("Kurunegala",   7.4867,  80.3647),
        "10": ("Badulla",      6.9895,  81.0557),
        "11": ("Ratnapura",    6.6828,  80.3992),
        "12": ("Other city",   None,    None   ),
    }
    print("\n  📍 Where are you located right now?")
    print("  " + "-"*40)
    for k, (name, _, _) in cities.items():
        print(f"     {k:>2}.  {name}")
    print("  " + "-"*40)
    while True:
        choice = input("  Type the number for your city: ").strip()
        if choice in cities:
            name, lat, lon = cities[choice]
            if choice == "12":
                name = input("  Type your city name: ").strip()
            return name, lat, lon
        print("  ⚠  Please type a number from the list above.")

def get_weather(city_name):
    try:
        API_KEY = "d36f2091f62593505c35c8451d609aa8"
        r = requests.get(
            f"http://api.openweathermap.org/data/2.5/weather"
            f"?q={city_name}&appid={API_KEY}&units=metric",
            timeout=5
        ).json()
        temp     = r["main"]["temp"]
        humidity = r["main"]["humidity"]
        pressure = r["main"]["pressure"]
        desc     = r["weather"][0]["description"]
        print(f"\n  🌤  Weather in {city_name}: {temp}°C, "
              f"{humidity}% humidity — {desc}")
        return temp, humidity, pressure
    except:
        print(f"\n  (Could not get live weather — "
              f"using typical values for {city_name})")
        defaults = {
            "Colombo":     (30, 78, 1010),
            "Kandy":       (26, 72, 1012),
            "Galle":       (29, 80, 1009),
            "Jaffna":      (33, 65, 1008),
            "Matara":      (28, 81, 1009),
            "Negombo":     (30, 77, 1010),
            "Badulla":     (22, 70, 1015),
            "Ratnapura":   (28, 82, 1009),
            "Trincomalee": (32, 68, 1007),
            "Anuradhapura":(31, 67, 1008),
            "Kurunegala":  (29, 73, 1010),
        }
        return defaults.get(city_name, (29, 75, 1010))

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
        print("\n  🟢 LOW RISK — You are doing well!")
        print("  " + "─"*44)
        print("  • Keep up your current routine")
        print("  • Stay hydrated — drink water regularly")
        print("  • Light walking is good for your joints")
        print("  • Take your medicines as scheduled")
        if humidity > 80:
            print(f"\n  💧 Humidity is a bit high in {city_name} today.")
            print("     Stay in cool areas when possible.")

    print("\n" + "="*58)

# ── Main Loop ─────────────────────────────────────────────────────────────────
print("\n  ✅  RA FLARE PREDICTION SYSTEM — Ready")
print("  Simple questions only — no medical knowledge needed.\n")

while True:
    collect_and_predict()
    again = input("\n  Assess another patient? (yes / no): ").strip().lower()
    if again != 'yes':
        print("\n  Thank you. Take care and stay well. 💙\n")
        break