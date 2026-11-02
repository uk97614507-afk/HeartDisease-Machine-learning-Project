import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
   
   
    page_title="Heart Disease Predicter by Uttam Kumar",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
}

.title {
    font-size: 38px;
    font-weight: bold;
    color: #e63946;
    text-align: center;
}

.subtitle {
    text-align: center;
    color: gray;
    font-size: 17px;
}

.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    model = joblib.load("KNN_heart.pkl")
    scaler = joblib.load("scaler.pkl")
    expected_columns = joblib.load("columns.pkl")

    return model, scaler, expected_columns


try:

    model, scaler, expected_columns = load_model()

except Exception as e:

    st.error("Model files load nahi ho paayi hain.")
    st.code(str(e))

    st.stop()


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("❤️ Heart Health")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Prediction",
        "About Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Developed by Uttam Kumar"
)


# ==========================================
# HOME PAGE
# ==========================================

if page == "Home":

    st.markdown(
        '<p class="title">❤️ Heart Disease Predicter</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">Machine Learning Based Prediction Application</p>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Welcome!")

        st.write("""
        Welcome to Heart Disease Predicter.

        This application uses a trained Machine Learning
        model to generate an educational prediction based
        on the health-related input features.
        """)

        st.success("Model loaded successfully!")

    with col2:

        st.subheader("Project Information")

        st.metric(
            label="Machine Learning Algorithm",
            value="KNN"
        )

        st.metric(
            label="Application",
            value="Streamlit"
        )

    st.markdown("---")

    st.subheader("Application Features")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.info("❤️ Heart Disease Prediction")

    with c2:
        st.info("📊 Probability Estimate")

    with c3:
        st.info("📋 Patient Input Summary")

    st.warning(
        "Educational project only. This application "
        "is not a medical diagnostic tool."
    )


# ==========================================
# PREDICTION PAGE
# ==========================================

elif page == "Prediction":

    st.title("❤️ Heart Disease Prediction")

    st.write(
        "Enter the required information below."
    )

    st.markdown("---")

    with st.form("prediction_form"):

        st.subheader("1. Personal Information")

        col1, col2 = st.columns(2)

        with col1:

            age = st.slider(
                "Age",
                18,
                100,
                40
            )

        with col2:

            sex = st.selectbox(
                "Sex",
                ["M", "F"]
            )

        st.markdown("---")

        st.subheader("2. Heart Related Information")

        col1, col2 = st.columns(2)

        with col1:

            chest_pain = st.selectbox(
                "Chest Pain Type",
                ["ATA", "NAP", "TA", "ASY"]
            )

            resting_bp = st.number_input(
                "Resting Blood Pressure",
                min_value=80,
                max_value=220,
                value=120
            )

            cholesterol = st.number_input(
                "Cholesterol",
                min_value=100,
                max_value=600,
                value=200
            )

            fasting_bs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dL",
                [0, 1]
            )

        with col2:

            resting_ecg = st.selectbox(
                "Resting ECG",
                ["Normal", "ST", "LVH"]
            )

            max_hr = st.slider(
                "Maximum Heart Rate",
                60,
                220,
                150
            )

            exercise_angina = st.selectbox(
                "Exercise Induced Angina",
                ["N", "Y"]
            )

            oldpeak = st.slider(
                "Oldpeak",
                -3.0,
                7.0,
                0.0,
                step=0.1
            )

        st.markdown("---")

        st.subheader("3. ST Slope Information")

        st_slope = st.selectbox(
            "ST Slope",
            ["Up", "Flat", "Down"]
        )

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:

            predict_button = st.form_submit_button(
                "🔍 Predict",
                use_container_width=True
            )

        with col2:

            reset_button = st.form_submit_button(
                "🔄 Reset",
                use_container_width=True
            )


    # ======================================
    # PREDICTION LOGIC
    # ======================================

    if reset_button:

        st.rerun()

    if predict_button:

        try:

            # Raw input dictionary

            raw_input = {

                "Age": age,
                "RestingBP": resting_bp,
                "Cholesterol": cholesterol,
                "FastingBS": fasting_bs,
                "MaxHR": max_hr,
                "Oldpeak": oldpeak,

                "Sex_" + sex: 1,

                "ChestPainType_" + chest_pain: 1,

                "RestingECG_" + resting_ecg: 1,

                "ExerciseAngina_" + exercise_angina: 1,

                "ST_Slope_" + st_slope: 1

            }

            # Convert into dataframe

            input_df = pd.DataFrame([raw_input])

            # Match training columns exactly

            input_df = input_df.reindex(
                columns=expected_columns,
                fill_value=0
            )

            # Scale input

            scaled_input = scaler.transform(input_df)

            # Prediction

            prediction = model.predict(scaled_input)[0]

            st.markdown("---")

            st.subheader("Prediction Result")

            if prediction == 1:

                st.error(
                    "⚠️ Model ne Heart Disease class predict ki hai."
                )

            else:

                st.success(
                    "✅ Model ne No Heart Disease class predict ki hai."
                )

            # Probability

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    scaled_input
                )[0]

                classes = list(model.classes_)

                if 1 in classes:

                    disease_probability = probabilities[
                        classes.index(1)
                    ] * 100

                    st.subheader("Model Probability Estimate")

                    st.metric(
                        "Heart Disease Class Probability",
                        f"{disease_probability:.2f}%"
                    )

                    st.progress(
                        int(round(disease_probability))
                    )

                    st.caption(
                        "This is the model's estimated class probability, "
                        "not a person's actual medical risk."
                    )

            # Patient summary

            st.markdown("---")

            st.subheader("Patient Information Summary")

            summary = pd.DataFrame({

                "Feature": [
                    "Age",
                    "Sex",
                    "Chest Pain",
                    "Resting BP",
                    "Cholesterol",
                    "Fasting Blood Sugar",
                    "Resting ECG",
                    "Maximum Heart Rate",
                    "Exercise Angina",
                    "Oldpeak",
                    "ST Slope"
                ],

                "Entered Value": [

                    age,
                    sex,
                    chest_pain,
                    resting_bp,
                    cholesterol,
                    fasting_bs,
                    resting_ecg,
                    max_hr,
                    exercise_angina,
                    oldpeak,
                    st_slope

                ]

            })

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True
            )

            st.warning(
                "This result is generated by a machine learning "
                "model for educational purposes only. It cannot "
                "diagnose or rule out heart disease. Consult a "
                "qualified healthcare professional for medical advice."
            )

        except Exception as e:

            st.error("Prediction ke time error aaya.")

            st.code(str(e))


# ==========================================
# ABOUT PROJECT PAGE
# ==========================================

elif page == "About Project":

    st.title("📚 About This Project")

    st.markdown("---")

    st.subheader("Project Name")

    st.write(
        "Heart Disease Predicter by Uttam Kumar"
    )

    st.subheader("Technology Used")

    technologies = [

        "Python",
        "Machine Learning",
        "K-Nearest Neighbors (KNN)",
        "Pandas",
        "NumPy",
        "Scikit-learn",
        "Streamlit",
        "Joblib"

    ]

    for tech in technologies:

        st.write("✔️", tech)

    st.markdown("---")

    st.subheader("How It Works")

    st.write("""
    1. User enters health-related information.

    2. Application converts the information into
       the required numerical features.

    3. Saved scaler transforms the input.

    4. Trained KNN model generates a prediction.

    5. Application displays the model output.
    """)

    st.markdown("---")

    st.subheader("Developer")

    st.success(
        "Developed by Uttam Kumar ❤️"
    )

    st.caption(
        "Educational and research project. Not intended "
        "for clinical diagnosis or treatment decisions."
    )


# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown(
    "<center>❤️ Heart Disease Predicter by Uttam Kumar</center>",
    unsafe_allow_html=True
)

