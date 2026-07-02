from pathlib import Path
import joblib
import pandas as pd
import streamlit as st


# =========================================================
# Paths
# =========================================================
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "clean" / "heart_big_data.csv"

MODEL_PATH = BASE_DIR / "model" / "heart_model.pkl"
THRESHOLD_PATH = BASE_DIR / "model" / "threshold.txt"
FEATURES_PATH = BASE_DIR / "model" / "features.txt"
MODEL_INFO_PATH = BASE_DIR / "model" / "model_info.txt"

GRAPH_DIR = BASE_DIR / "graphs"
RESULT_DIR = BASE_DIR / "results"
BIG_RESULT_PATH = RESULT_DIR / "big_model_results.csv"
SHAP_FEATURES_PATH = RESULT_DIR / "shap_features.csv"


# =========================================================
# Page Config
# =========================================================
st.set_page_config(
    page_title="Heart Attack Risk Analyzer",
    page_icon="❤️",
    layout="wide"
)


# =========================================================
# CSS Styling
# =========================================================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #fff7f7 0%, #f8fafc 45%, #eef2ff 100%);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #1f2937 100%);
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    .hero-card {
        padding: 32px;
        border-radius: 26px;
        background: linear-gradient(135deg, #ef4444, #be123c);
        color: white;
        box-shadow: 0px 18px 45px rgba(239, 68, 68, 0.25);
        margin-bottom: 28px;
    }

    .hero-title {
        font-size: 44px;
        font-weight: 900;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 18px;
        opacity: 0.95;
        line-height: 1.6;
    }

    .glass-card {
        padding: 24px;
        border-radius: 22px;
        background: rgba(255, 255, 255, 0.90);
        border: 1px solid rgba(255, 255, 255, 0.75);
        box-shadow: 0px 10px 30px rgba(15, 23, 42, 0.08);
        margin-bottom: 22px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 850;
        color: #111827;
        margin-bottom: 6px;
    }

    .section-subtitle {
        color: #6b7280;
        font-size: 15px;
        margin-bottom: 20px;
    }

    div[data-testid="stMetric"] {
        background: white;
        padding: 18px;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        box-shadow: 0px 8px 22px rgba(15, 23, 42, 0.06);
    }

    .risk-box-high {
        padding: 24px;
        border-radius: 22px;
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        border: 1px solid #ef4444;
        color: #7f1d1d;
        font-size: 20px;
        font-weight: 800;
        margin-bottom: 18px;
    }

    .risk-box-medium {
        padding: 24px;
        border-radius: 22px;
        background: linear-gradient(135deg, #fef3c7, #fde68a);
        border: 1px solid #f59e0b;
        color: #78350f;
        font-size: 20px;
        font-weight: 800;
        margin-bottom: 18px;
    }

    .risk-box-low {
        padding: 24px;
        border-radius: 22px;
        background: linear-gradient(135deg, #dcfce7, #bbf7d0);
        border: 1px solid #22c55e;
        color: #14532d;
        font-size: 20px;
        font-weight: 800;
        margin-bottom: 18px;
    }

    .info-pill {
        display: inline-block;
        padding: 8px 14px;
        border-radius: 999px;
        background: #fef2f2;
        border: 1px solid #fecaca;
        color: #991b1b;
        font-weight: 700;
        margin-right: 8px;
        margin-bottom: 8px;
    }

    .good-pill {
        display: inline-block;
        padding: 8px 14px;
        border-radius: 999px;
        background: #ecfdf5;
        border: 1px solid #bbf7d0;
        color: #166534;
        font-weight: 700;
        margin-right: 8px;
        margin-bottom: 8px;
    }

    .stButton > button {
        border-radius: 14px;
        padding: 0.75rem 1.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ef4444, #be123c);
        color: white;
        border: none;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #dc2626, #9f1239);
        color: white;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# Helper Functions
# =========================================================
@st.cache_data(show_spinner=False)
def load_target_data():
    return pd.read_csv(DATA_PATH, usecols=["heart_attack", "year"])


@st.cache_data(show_spinner=False)
def load_feature_data(feature_name):
    return pd.read_csv(DATA_PATH, usecols=[feature_name, "heart_attack"])


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)


def load_threshold():
    if THRESHOLD_PATH.exists():
        return float(THRESHOLD_PATH.read_text().strip())
    return 0.50


def load_features():
    if FEATURES_PATH.exists():
        return [
            line.strip()
            for line in FEATURES_PATH.read_text().splitlines()
            if line.strip()
        ]
    return []


def feature_label(feature):
    labels = {
        "state": "Location / State",
        "year": "Survey Year",
        "age_group": "Age Group",
        "sex": "Sex",
        "general_health": "General Health",
        "physical_bad_days": "Poor Physical Health Days",
        "mental_bad_days": "Poor Mental Health Days",
        "bmi": "BMI",
        "physical_activity": "Physical Activity",
        "smoking_status": "Smoking Status",
        "heavy_drinking": "Heavy Drinking",
        "sleep_hours": "Sleep Hours",
        "diabetes": "Diabetes",
        "stroke": "Stroke History",
        "kidney_disease": "Kidney Disease",
        "asthma": "Asthma",
        "copd": "COPD",
        "depression": "Depression",
        "arthritis": "Arthritis",
        "high_bp": "High Blood Pressure",
        "high_cholesterol": "High Cholesterol",
        "insurance": "Insurance",
        "personal_doctor": "Personal Doctor",
        "cost_barrier": "Medical Cost Barrier",
        "checkup": "Routine Checkup",
    }
    return labels.get(feature, feature)


def map_labels(feature, series):
    maps = {
        "sex": {1: "Male", 2: "Female"},
        "general_health": {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"},
        "physical_activity": {1: "Yes", 2: "No"},
        "smoking_status": {1: "Every day smoker", 2: "Some days smoker", 3: "Former smoker", 4: "Never smoked"},
        "heavy_drinking": {1: "No", 2: "Yes"},
        "diabetes": {1: "Yes", 2: "During pregnancy", 3: "No", 4: "Prediabetes"},
        "stroke": {1: "Yes", 2: "No"},
        "kidney_disease": {1: "Yes", 2: "No"},
        "asthma": {1: "Yes", 2: "No"},
        "copd": {1: "Yes", 2: "No"},
        "depression": {1: "Yes", 2: "No"},
        "arthritis": {1: "Yes", 2: "No"},
        "high_bp": {1: "Yes", 2: "During pregnancy", 3: "No", 4: "Borderline"},
        "high_cholesterol": {1: "Yes", 2: "No"},
        "personal_doctor": {1: "One doctor", 2: "More than one", 3: "No"},
        "cost_barrier": {1: "Yes", 2: "No"},
        "checkup": {1: "Within 1 year", 2: "Within 2 years", 3: "Within 5 years", 4: "5+ years ago", 8: "Never"},
    }

    if feature in maps:
        return series.map(maps[feature]).fillna(series.astype(str))

    return series.astype(str)


def risk_level(score):
    if score < 30:
        return "Low Risk"
    if score < 60:
        return "Medium Risk"
    return "High Risk"


def build_positive_negative_factors(input_data):
    increasing = []
    reducing = []

    if input_data["age_group"] >= 10:
        increasing.append("Higher age group")
    else:
        reducing.append("Lower age group")

    if input_data["general_health"] >= 4:
        increasing.append("Fair/poor general health")
    elif input_data["general_health"] <= 2:
        reducing.append("Excellent/very good general health")

    if input_data["bmi"] >= 30:
        increasing.append("BMI in obesity range")
    elif 18.5 <= input_data["bmi"] < 25:
        reducing.append("BMI in normal range")

    if input_data["smoking_status"] in [1, 2]:
        increasing.append("Current smoking")
    elif input_data["smoking_status"] == 4:
        reducing.append("Never smoked")

    if input_data["physical_activity"] == 2:
        increasing.append("No physical activity in past month")
    else:
        reducing.append("Physical activity present")

    if input_data["diabetes"] in [1, 4]:
        increasing.append("Diabetes or prediabetes")

    if input_data["stroke"] == 1:
        increasing.append("Previous stroke history")

    if input_data["kidney_disease"] == 1:
        increasing.append("Kidney disease")

    if input_data["copd"] == 1:
        increasing.append("COPD / chronic lung condition")

    if input_data["high_bp"] in [1, 4]:
        increasing.append("High or borderline blood pressure")

    if input_data["high_cholesterol"] == 1:
        increasing.append("High cholesterol")

    if input_data["sleep_hours"] < 6:
        increasing.append("Low sleep duration")
    elif 7 <= input_data["sleep_hours"] <= 9:
        reducing.append("Healthy sleep duration")

    if input_data["checkup"] == 1:
        reducing.append("Recent routine checkup")
    elif input_data["checkup"] in [4, 8]:
        increasing.append("No recent routine checkup")

    if not increasing:
        increasing.append("No major high-risk pattern selected")

    if not reducing:
        reducing.append("No strong protective pattern selected")

    return increasing, reducing


def build_guidance(input_data, systolic_bp, diastolic_bp):
    guidance = []

    if input_data["smoking_status"] in [1, 2]:
        guidance.append("Consider smoking cessation support and consult a healthcare professional.")

    if input_data["bmi"] >= 30:
        guidance.append("Work on gradual weight management through diet and safe physical activity.")

    if input_data["physical_activity"] == 2:
        guidance.append("Add regular physical activity if medically safe. Start slowly if inactive.")

    if input_data["high_bp"] in [1, 4] or systolic_bp >= 130 or diastolic_bp >= 80:
        guidance.append("Monitor blood pressure regularly and follow medical advice.")

    if input_data["high_cholesterol"] == 1:
        guidance.append("Monitor cholesterol and discuss lipid control with a doctor.")

    if input_data["diabetes"] in [1, 4]:
        guidance.append("Manage blood sugar through regular checkups, diet, and prescribed treatment.")

    if input_data["general_health"] >= 4:
        guidance.append("Schedule a full health checkup to review overall health status.")

    if input_data["sleep_hours"] < 6:
        guidance.append("Improve sleep routine where possible.")

    if not guidance:
        guidance.append("Maintain regular checkups, balanced diet, physical activity, and healthy sleep.")

    return guidance


def show_page_header(title, subtitle):
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="section-title">{title}</div>
            <div class="section-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# Load Model Assets
# =========================================================
model = None
threshold = 0.50
features = []

if MODEL_PATH.exists():
    model = load_model()

threshold = load_threshold()
features = load_features()


# =========================================================
# Sidebar Navigation
# =========================================================
with st.sidebar:
    st.markdown("## ❤️ HeartAI")
    st.caption("Explainable heart attack risk analyzer")

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "📈 Graph Analysis",
            "🫀 Predict Risk",
            "🔍 Explainability",
            "⚠️ Disclaimer"
        ]
    )

    st.divider()
    st.caption("Model trained on CDC BRFSS multi-year data")
    st.caption("Research tool · Not medical diagnosis")


# =========================================================
# Hero Header
# =========================================================
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">❤️ Heart Attack Risk Analyzer</div>
        <div class="hero-subtitle">
            Explainable AI system trained on 48 lakh+ real-world CDC BRFSS health records.
            Analyze data, predict risk, and understand the factors behind the result.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# Page 1: Dashboard
# =========================================================
if page == "🏠 Dashboard":
    show_page_header(
        "📊 Project Dashboard",
        "Overview of dataset size, heart attack cases, yearly patterns, and model training details."
    )

    if not DATA_PATH.exists():
        st.error("Big dataset not found: data/clean/heart_big_data.csv")
        st.stop()

    target_df = load_target_data()

    total_rows = len(target_df)
    heart_cases = int(target_df["heart_attack"].sum())
    no_cases = total_rows - heart_cases
    heart_percent = (heart_cases / total_rows) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{total_rows:,}")
    c2.metric("Heart Attack Cases", f"{heart_cases:,}")
    c3.metric("No Heart Attack Cases", f"{no_cases:,}")
    c4.metric("Heart Attack Rate", f"{heart_percent:.2f}%")

    st.divider()

    year_summary = (
        target_df.groupby("year")["heart_attack"]
        .agg(total_records="count", heart_attack_cases="sum")
        .reset_index()
    )

    year_summary["heart_attack_rate_%"] = (
        year_summary["heart_attack_cases"] / year_summary["total_records"] * 100
    )

    left, right = st.columns([1.5, 1])

    with left:
        st.subheader("Year-wise Heart Attack Rate")
        st.line_chart(year_summary.set_index("year")["heart_attack_rate_%"])

    with right:
        st.subheader("Class Distribution")
        class_df = pd.DataFrame(
            {
                "Class": ["No Heart Attack", "Heart Attack"],
                "Count": [no_cases, heart_cases]
            }
        )
        st.bar_chart(class_df.set_index("Class"))

    with st.expander("Show full year-wise dataset details"):
        st.dataframe(year_summary, use_container_width=True)

    st.divider()

    st.subheader("Model Training Details")

    m1, m2 = st.columns(2)

    with m1:
        if MODEL_INFO_PATH.exists():
            with st.expander("Show model info"):
                st.code(MODEL_INFO_PATH.read_text())
        else:
            st.info("model/model_info.txt not found.")

    with m2:
        if BIG_RESULT_PATH.exists():
            result_df = pd.read_csv(BIG_RESULT_PATH)
            best_rows = result_df.sort_values("f1_score", ascending=False).head(10)
            with st.expander("Show top model results"):
                st.dataframe(best_rows, use_container_width=True)
        else:
            st.info("results/big_model_results.csv not found.")


# =========================================================
# Page 2: Graph Analysis
# =========================================================
elif page == "📈 Graph Analysis":
    show_page_header(
        "📈 Ask Graph Analysis",
        "Choose any health, lifestyle, location, or disease factor and generate analysis from the large dataset."
    )

    available_features = [
        "year",
        "state",
        "age_group",
        "sex",
        "general_health",
        "bmi",
        "physical_activity",
        "smoking_status",
        "heavy_drinking",
        "sleep_hours",
        "diabetes",
        "stroke",
        "kidney_disease",
        "asthma",
        "copd",
        "depression",
        "arthritis",
        "high_bp",
        "high_cholesterol",
        "personal_doctor",
        "cost_barrier",
        "checkup",
    ]

    col1, col2, col3 = st.columns([1.2, 1, 1])

    selected_feature = col1.selectbox(
        "What do you want to analyze?",
        available_features,
        format_func=feature_label
    )

    analysis_type = col2.selectbox(
        "Analysis Type",
        ["Heart attack rate", "Record count", "Both"]
    )

    min_records = col3.number_input(
        "Minimum records per group",
        min_value=1,
        max_value=100000,
        value=1000,
        step=1000
    )

    if st.button("Generate Analysis", type="primary"):
        with st.spinner("Analyzing large dataset..."):
            temp = load_feature_data(selected_feature)

            if selected_feature == "bmi":
                temp["group"] = pd.cut(
                    temp[selected_feature],
                    bins=[0, 18.5, 25, 30, 35, 100],
                    labels=["Underweight", "Normal", "Overweight", "Obese I", "Obese II+"]
                )

            elif selected_feature == "sleep_hours":
                temp["group"] = pd.cut(
                    temp[selected_feature],
                    bins=[0, 5, 6, 8, 10, 24],
                    labels=["≤5 hrs", "6 hrs", "7–8 hrs", "9–10 hrs", "10+ hrs"]
                )

            elif selected_feature in ["physical_bad_days", "mental_bad_days"]:
                temp["group"] = pd.cut(
                    temp[selected_feature],
                    bins=[-1, 0, 5, 15, 30],
                    labels=["0 days", "1–5 days", "6–15 days", "16–30 days"]
                )

            else:
                temp["group"] = map_labels(selected_feature, temp[selected_feature])

            summary = (
                temp.groupby("group", observed=False)["heart_attack"]
                .agg(total_records="count", heart_attack_cases="sum")
                .reset_index()
            )

            summary = summary[summary["total_records"] >= min_records].copy()

            summary["heart_attack_rate_%"] = (
                summary["heart_attack_cases"] / summary["total_records"] * 100
            )

            summary = summary.sort_values("heart_attack_rate_%", ascending=False)

            st.subheader(f"Analysis by {feature_label(selected_feature)}")

            if summary.empty:
                st.warning("No groups found after applying minimum record filter.")
            else:
                top_row = summary.iloc[0]

                a1, a2, a3 = st.columns(3)
                a1.metric("Groups Analyzed", len(summary))
                a2.metric("Highest Rate Group", str(top_row["group"]))
                a3.metric("Highest Rate", f"{top_row['heart_attack_rate_%']:.2f}%")

                st.divider()

                if analysis_type in ["Heart attack rate", "Both"]:
                    st.subheader("Heart Attack Rate by Group")
                    st.bar_chart(summary.set_index("group")["heart_attack_rate_%"])

                if analysis_type in ["Record count", "Both"]:
                    st.subheader("Record Count by Group")
                    st.bar_chart(summary.set_index("group")["total_records"])

                with st.expander("Show detailed table"):
                    st.dataframe(summary, use_container_width=True)


# =========================================================
# Page 3: Predict Risk
# =========================================================
elif page == "🫀 Predict Risk":
    show_page_header(
        "🫀 Predict Heart Attack Risk",
        "Enter health, lifestyle, and previous disease details. The model returns risk score and explanation."
    )

    if model is None:
        st.error("Model file not found: model/heart_model.pkl")
        st.stop()

    if not features:
        st.error("Feature file not found: model/features.txt")
        st.stop()

    with st.expander("👤 Basic Details", expanded=True):
        b1, b2, b3, b4 = st.columns(4)

        state = b1.number_input("State / Location Code", min_value=1, max_value=78, value=6)
        year = b2.number_input("Reference Year", min_value=2011, max_value=2024, value=2024)

        age_group = b3.selectbox(
            "Age Group",
            [
                ("18–24", 1), ("25–29", 2), ("30–34", 3), ("35–39", 4),
                ("40–44", 5), ("45–49", 6), ("50–54", 7), ("55–59", 8),
                ("60–64", 9), ("65–69", 10), ("70–74", 11), ("75–79", 12),
                ("80+", 13)
            ],
            format_func=lambda x: x[0]
        )[1]

        sex = b4.selectbox(
            "Sex",
            [("Male", 1), ("Female", 2)],
            format_func=lambda x: x[0]
        )[1]

    with st.expander("🧬 Current Health", expanded=True):
        h1, h2, h3, h4 = st.columns(4)

        general_health = h1.selectbox(
            "General Health",
            [("Excellent", 1), ("Very good", 2), ("Good", 3), ("Fair", 4), ("Poor", 5)],
            format_func=lambda x: x[0]
        )[1]

        bmi = h2.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0, step=0.1)
        sleep_hours = h3.slider("Sleep Hours", 1, 24, 7)

        physical_activity = h4.selectbox(
            "Physical Activity?",
            [("Yes", 1), ("No", 2)],
            format_func=lambda x: x[0]
        )[1]

        h5, h6, h7, h8 = st.columns(4)

        physical_bad_days = h5.slider("Poor Physical Health Days", 0, 30, 0)
        mental_bad_days = h6.slider("Poor Mental Health Days", 0, 30, 0)

        systolic_bp = h7.number_input("Current Systolic BP", min_value=70, max_value=250, value=120)
        diastolic_bp = h8.number_input("Current Diastolic BP", min_value=40, max_value=160, value=80)

    with st.expander("🚬 Lifestyle", expanded=True):
        l1, l2 = st.columns(2)

        smoking_status = l1.selectbox(
            "Smoking Status",
            [
                ("Every day smoker", 1),
                ("Some days smoker", 2),
                ("Former smoker", 3),
                ("Never smoked", 4)
            ],
            format_func=lambda x: x[0]
        )[1]

        heavy_drinking = l2.selectbox(
            "Heavy Drinking",
            [("No", 1), ("Yes", 2)],
            format_func=lambda x: x[0]
        )[1]

    with st.expander("🏥 Previous Diseases", expanded=True):
        d1, d2, d3, d4 = st.columns(4)

        diabetes = d1.selectbox(
            "Diabetes",
            [("Yes", 1), ("During pregnancy", 2), ("No", 3), ("Prediabetes", 4)],
            format_func=lambda x: x[0]
        )[1]

        stroke = d2.selectbox("Stroke History", [("Yes", 1), ("No", 2)], format_func=lambda x: x[0])[1]
        kidney_disease = d3.selectbox("Kidney Disease", [("Yes", 1), ("No", 2)], format_func=lambda x: x[0])[1]
        asthma = d4.selectbox("Asthma", [("Yes", 1), ("No", 2)], format_func=lambda x: x[0])[1]

        d5, d6, d7, d8 = st.columns(4)

        copd = d5.selectbox("COPD", [("Yes", 1), ("No", 2)], format_func=lambda x: x[0])[1]
        depression = d6.selectbox("Depression", [("Yes", 1), ("No", 2)], format_func=lambda x: x[0])[1]
        arthritis = d7.selectbox("Arthritis", [("Yes", 1), ("No", 2)], format_func=lambda x: x[0])[1]
        high_cholesterol = d8.selectbox("High Cholesterol", [("Yes", 1), ("No", 2)], format_func=lambda x: x[0])[1]

        high_bp = st.selectbox(
            "High Blood Pressure",
            [("Yes", 1), ("During pregnancy", 2), ("No", 3), ("Borderline", 4)],
            format_func=lambda x: x[0]
        )[1]

    with st.expander("🩺 Healthcare Access", expanded=False):
        a1, a2, a3, a4 = st.columns(4)

        insurance = a1.selectbox(
            "Insurance",
            [("Yes / Available", 1), ("No coverage", 10)],
            format_func=lambda x: x[0]
        )[1]

        personal_doctor = a2.selectbox(
            "Personal Doctor",
            [("One", 1), ("More than one", 2), ("No", 3)],
            format_func=lambda x: x[0]
        )[1]

        cost_barrier = a3.selectbox(
            "Could not see doctor due to cost?",
            [("Yes", 1), ("No", 2)],
            format_func=lambda x: x[0]
        )[1]

        checkup = a4.selectbox(
            "Last Routine Checkup",
            [
                ("Within 1 year", 1),
                ("Within 2 years", 2),
                ("Within 5 years", 3),
                ("5+ years ago", 4),
                ("Never", 8)
            ],
            format_func=lambda x: x[0]
        )[1]

    input_data = {
        "state": state,
        "year": year,
        "age_group": age_group,
        "sex": sex,
        "general_health": general_health,
        "physical_bad_days": physical_bad_days,
        "mental_bad_days": mental_bad_days,
        "bmi": bmi,
        "physical_activity": physical_activity,
        "smoking_status": smoking_status,
        "heavy_drinking": heavy_drinking,
        "sleep_hours": sleep_hours,
        "diabetes": diabetes,
        "stroke": stroke,
        "kidney_disease": kidney_disease,
        "asthma": asthma,
        "copd": copd,
        "depression": depression,
        "arthritis": arthritis,
        "high_bp": high_bp,
        "high_cholesterol": high_cholesterol,
        "insurance": insurance,
        "personal_doctor": personal_doctor,
        "cost_barrier": cost_barrier,
        "checkup": checkup,
    }

    final_input = pd.DataFrame([input_data])

    for col in features:
        if col not in final_input.columns:
            final_input[col] = 0

    final_input = final_input[features]

    st.divider()

    if st.button("Predict Risk", type="primary"):
        probability = model.predict_proba(final_input)[0][1]
        risk_percent = probability * 100
        level = risk_level(risk_percent)

        r1, r2, r3 = st.columns(3)
        r1.metric("Estimated Risk Score", f"{risk_percent:.2f}%")
        r2.metric("Risk Category", level)
        r3.metric("Model Threshold", f"{threshold:.2f}")

        if level == "High Risk":
            st.markdown(
                """
                <div class="risk-box-high">
                    🔴 High Risk Pattern Detected<br>
                    Please consider medical consultation and lifestyle risk review.
                </div>
                """,
                unsafe_allow_html=True
            )
        elif level == "Medium Risk":
            st.markdown(
                """
                <div class="risk-box-medium">
                    🟠 Medium Risk Pattern Detected<br>
                    Some risk factors are present. Improvement and checkup are recommended.
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div class="risk-box-low">
                    🟢 Low Risk Pattern Detected<br>
                    Current selected inputs show comparatively lower risk pattern.
                </div>
                """,
                unsafe_allow_html=True
            )

        increasing, reducing = build_positive_negative_factors(input_data)

        st.subheader("Why this prediction is showing")

        inc_col, red_col = st.columns(2)

        with inc_col:
            st.write("### 🔺 Factors increasing estimated risk")
            for item in increasing:
                st.markdown(f"<span class='info-pill'>{item}</span>", unsafe_allow_html=True)

        with red_col:
            st.write("### 🟢 Factors reducing estimated risk")
            for item in reducing:
                st.markdown(f"<span class='good-pill'>{item}</span>", unsafe_allow_html=True)

        st.subheader("Risk-reduction guidance")

        guidance = build_guidance(input_data, systolic_bp, diastolic_bp)

        for item in guidance:
            st.write(f"- {item}")

        st.info("This is not a cure or diagnosis. It is risk-reduction guidance based on public-health patterns.")


# =========================================================
# Page 4: Explainability
# =========================================================
elif page == "🔍 Explainability":
    show_page_header(
        "🔍 Explainability",
        "SHAP explains how the model understands important heart attack risk-related factors."
    )

    shap_bar = GRAPH_DIR / "shap_bar.png"
    shap_beeswarm = GRAPH_DIR / "shap_beeswarm.png"
    shap_waterfall = GRAPH_DIR / "shap_waterfall.png"

    if shap_bar.exists():
        st.subheader("Overall Feature Importance")
        st.image(str(shap_bar), use_container_width=True)
    else:
        st.info("SHAP bar plot not found. Run scripts/shap_explain.py.")

    if shap_beeswarm.exists():
        st.subheader("Feature Impact Direction")
        st.image(str(shap_beeswarm), use_container_width=True)

    if shap_waterfall.exists():
        st.subheader("Single Prediction Explanation")
        st.image(str(shap_waterfall), use_container_width=True)

    if SHAP_FEATURES_PATH.exists():
        st.subheader("Top SHAP Features")
        shap_df = pd.read_csv(SHAP_FEATURES_PATH)
        st.dataframe(shap_df.head(15), use_container_width=True)

    st.warning("SHAP explains model behavior. It does not prove medical causation.")


# =========================================================
# Page 5: Disclaimer
# =========================================================
elif page == "⚠️ Disclaimer":
    show_page_header(
        "⚠️ Important Disclaimer",
        "This section explains what the project can and cannot do."
    )

    st.error(
        """
        If someone has chest pain, shortness of breath, sweating, fainting,
        pain spreading to arm/jaw/back, or severe discomfort, seek emergency
        medical help immediately.
        """
    )

    st.subheader("What this project does")

    st.write(
        """
        - Analyzes real-world CDC BRFSS health survey data
        - Studies lifestyle and previous disease patterns
        - Trains a machine learning model
        - Estimates risk category
        - Explains important factors using SHAP
        """
    )

    st.subheader("What this project does not do")

    st.write(
        """
        - It does not diagnose heart attack
        - It does not confirm a heart condition
        - It does not provide a cure
        - It does not replace clinical tests
        - It does not replace a doctor
        - It does not predict exact future heart attack events
        """
    )