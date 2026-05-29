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
        API_KEY = "Add api key"
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

def safe_prob(model, inputs):
    n   = model.n_features_in_
    arr = np.array(inputs, dtype=float)
    if len(arr) < n:
        arr = np.pad(arr, (0, n - len(arr)))
    else:
        arr = arr[:n]
    return float(model.predict_proba([arr])[0][1])

def normalize(val, lo, hi):
    return float(np.clip((val - lo) / (hi - lo), 0, 1))

def collect_and_predict():
    print("\n" + "="*58)
    print("   🏥  RA FLARE RISK PREDICTION")
    print("="*58)

    # ── Name (required) ───────────────────────────────────────────────────────
    name = get_name()

    # ── Location + Weather ────────────────────────────────────────────────────
    city_name, lat, lon = choose_location()
    temp, humidity, pressure = get_weather(city_name)

    # ── Section 1: Joint Pain ─────────────────────────────────────────────────
    print("\n  " + "─"*50)
    print("  SECTION 1 — Your Joint Pain Today")
    print("  " + "─"*50)

    pain = ask(
        "How much pain are you feeling in your joints right now?\n"
        "  (0 = no pain at all  →  10 = unbearable pain)",
        0, 10
    )
    swollen = ask(
        "How many joints feel swollen or puffy to you?\n"
        "  You can count fingers, wrists, elbows, knees, ankles\n"
        "  (0 = none  →  28 = many joints)",
        0, 28
    )
    stiffness = ask(
        "When you woke up this morning, how long did your joints\n"
        "  feel stiff or hard to move before getting better?\n"
        "  (0 = no stiffness  →  12 = stiff for 12 hours or more)",
        0, 12
    )
    fatigue = ask(
        "How tired and worn-out do you feel overall today?\n"
        "  (0 = not tired at all  →  10 = completely exhausted)",
        0, 10
    )
    flares30 = ask(
        "In the past 30 days, how many times did your joint pain\n"
        "  suddenly get much worse all at once?\n"
        "  (0 = never  →  30 = almost every day)",
        0, 30
    )
    duration = ask(
        "How many years ago were you first told you have RA?\n"
        "  (0 = less than 1 year  →  50 = 50 years ago)",
        0, 50
    )

    # ── Section 2: Sleep & Stress ─────────────────────────────────────────────
    print("\n  " + "─"*50)
    print("  SECTION 2 — Your Sleep & Stress")
    print("  " + "─"*50)

    sleep_hrs = ask(
        "How many hours did you sleep last night?\n"
        "  (0 = did not sleep at all  →  12 = slept 12 hours)",
        0, 12
    )
    sleep_qual = ask(
        "How well did you sleep last night overall?\n"
        "  (1 = very badly, kept waking up  →  10 = slept perfectly)",
        1, 10
    )
    stress = ask(
        "How stressed or anxious are you feeling today?\n"
        "  (1 = very calm and relaxed  →  10 = extremely stressed)",
        1, 10
    )
    activity = ask(
        "How many minutes did you spend walking or doing\n"
        "  any light movement today?\n"
        "  (0 = none at all  →  180 = 3 hours or more)",
        0, 180
    )

    # ── Section 3: General Health ─────────────────────────────────────────────
    print("\n  " + "─"*50)
    print("  SECTION 3 — Your General Health Today")
    print("  " + "─"*50)

    heart_rate = ask(
        "What is your heart rate (pulse) right now?\n"
        "  → Check your phone health app or smartwatch\n"
        "  → Or count your pulse for 15 seconds × 4\n"
        "  (normal resting pulse = 60 to 100 beats per minute)",
        40, 180
    )
    energy = ask(
        "How is your energy level today overall?\n"
        "  (1 = feel completely drained  →  10 = feeling great)",
        1, 10
    )
    appetite = ask(
        "How is your appetite today?\n"
        "  (1 = cannot eat anything  →  10 = eating normally)",
        1, 10
    )

    # ── Build model inputs ────────────────────────────────────────────────────
    ra_in = [
        pain, swollen, stiffness, fatigue, flares30,
        duration, humidity, temp, pressure,
        1 if humidity > 75 else 0
    ]

    das28_proxy = round(
        0.56 * np.sqrt(max(swollen, 0)) +
        0.28 * np.sqrt(max(pain, 0)) +
        0.70 * np.log(max(fatigue, 1)) +
        0.014 * (10 - energy), 2
    )
    sl_ra_in = [
        das28_proxy, pain * 7.6, swollen, pain,
        fatigue, duration * 12, stiffness, 10 - energy
    ]

    sleep_in = [sleep_hrs, sleep_qual, stress, activity, heart_rate, energy]

    rmssd_est = max(10, 60 - (stress * 4) - (heart_rate * 0.2) + (sleep_qual * 2))
    sdnn_est  = rmssd_est * 1.3
    lf_hf_est = 1.0 + (stress * 0.3) - (sleep_qual * 0.1)
    hrv_in = [
        rmssd_est, rmssd_est, sdnn_est, rmssd_est,
        rmssd_est * 0.8,
        rmssd_est / max(sdnn_est, 1),
        heart_rate,
        max(0, 50 - stress * 3),
        max(0, 30 - stress * 2),
        rmssd_est * 0.7, sdnn_est * 1.2,
        stress * 0.5, (stress - 5) * 0.2, lf_hf_est,
    ]

    # ── Clinical risk score directly from patient answers ─────────────────────
    # Weighted sum — mirrors clinical RA assessment guidelines
    clinical_score = (
        normalize(pain,      0, 10) * 0.30 +   # pain = strongest signal
        normalize(swollen,   0, 28) * 0.20 +   # swollen joints
        normalize(flares30,  0, 30) * 0.20 +   # recent flare history
        normalize(fatigue,   0, 10) * 0.10 +   # fatigue
        normalize(stiffness, 0, 12) * 0.10 +   # morning stiffness
        normalize(stress,    1, 10) * 0.05 +   # stress
        (1 - normalize(sleep_hrs, 0, 12)) * 0.05  # poor sleep = more risk
    )

    # ML model signals
    p_ra    = np.clip(safe_prob(ra_model,    ra_in),    0, 1)
    p_sl    = np.clip(safe_prob(sl_ra_model, sl_ra_in), 0, 1)
    p_sleep = np.clip(safe_prob(sleep_model, sleep_in), 0, 1)
    p_hrv   = np.clip(safe_prob(hrv_model,   hrv_in),  0, 1)
    meta_score = float(
        meta_model.predict_proba([[p_ra, p_sleep, p_hrv, p_sl]])[0][1]
    )

    # Final combined risk: 60% clinical answers + 40% ML models
    base_risk = (clinical_score * 0.60) + (meta_score * 0.40)

    # Sri Lanka humidity boost
    if humidity > 80:
        base_risk = min(1.0, base_risk + 0.05)

    # ── 3-Day Prediction ──────────────────────────────────────────────────────
    print(f"\n\n  ⏳ Calculating 3-day flare risk for {name}...")
    print("="*58)

    day_risks = []
    for day in range(1, 4):
        drift = np.random.normal(0, 0.015) * day
        risk  = float(np.clip(base_risk + drift, 0, 1))
        day_risks.append(risk)

        label = ("🔴 HIGH RISK    " if risk > 0.65 else
                 "🟡 MODERATE RISK" if risk > 0.35 else
                 "🟢 LOW RISK     ")

        print(f"\n  Day +{day}:  {label}  (score: {risk:.2f})")

    # ── Summary & Advice ──────────────────────────────────────────────────────
    peak = max(day_risks)
    print("\n" + "="*58)
    print(f"  Patient  : {name}")
    print(f"  Location : {city_name}  |  {temp}°C  |  {humidity}% humidity")
    print(f"  Peak Risk: {peak:.2f}")
    print("="*58)

    if peak > 0.65:
        print("\n  🔴 HIGH FLARE RISK in the next 3 days")
        print("  " + "─"*44)
        print("  • Please contact your doctor or rheumatologist today")
        print("  • Rest as much as possible — avoid heavy lifting")
        print("  • Take your prescribed medicines on time")
        print("  • Apply warm or cold packs on painful joints")
        print("  • Avoid going outside during midday heat")
        if humidity > 75:
            print(f"\n  ⚠  Humidity is HIGH in {city_name} right now.")
            print("     High humidity worsens joint inflammation.")
            print("     Stay indoors and use a fan or AC if possible.")

    elif peak > 0.35:
        print("\n  🟡 MODERATE RISK — Please take care")
        print("  " + "─"*44)
        print("  • Try to sleep at least 7–8 hours tonight")
        print("  • Avoid stressful activities today")
        print("  • Drink plenty of water throughout the day")
        print("  • Do gentle stretching only — no heavy exercise")
        print("  • If your pain gets worse, call your doctor")
        if humidity > 75:
            print(f"\n  💧 Humidity is elevated in {city_name}.")
            print("     Take cool rest breaks and stay hydrated.")

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
