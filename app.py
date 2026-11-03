import streamlit as st
import requests

st.set_page_config(page_title="Student Mental Health Predictor", page_icon="🎓")

st.title("🎓 Student Mental Health Score Predictor")
st.write(
    "Enter the student's details below. The app sends them to the FastAPI "
    "backend and displays the predicted score."
)

API_URL = "http://127.0.0.1:8000/predict"

with st.form("student_form"):
    age = st.number_input("Age", min_value=10, max_value=100, value=20)
    gender = st.text_input("Gender", value="Male")
    country = st.text_input("Country", value="India")
    academic_level = st.text_input("Academic Level", value="Undergraduate")

    usage = st.number_input(
        "Average daily social media usage (hours)",
        min_value=0.0, max_value=24.0, value=3.0, step=0.5
    )
    platform = st.text_input("Most used platform", value="Instagram")
    unlocks = st.number_input(
        "Daily phone unlocks", min_value=0, value=50, step=1
    )
    study = st.number_input(
        "Study hours per day", min_value=0.0, max_value=24.0,
        value=3.0, step=0.5
    )
    sleep = st.number_input(
        "Sleep hours per night", min_value=0.0, max_value=24.0,
        value=7.0, step=0.5
    )
    activity = st.number_input(
        "Physical activity hours per day", min_value=0.0, max_value=24.0,
        value=1.0, step=0.5
    )
    stress = st.selectbox(
        "Stress level", ["Low", "Medium", "High", "Very High"]
    )
    purpose = st.text_input("Purpose of use", value="Entertainment")

    submitted = st.form_submit_button("Predict Mental Health Score")

if submitted:
    student_data = {
        "Age": age,
        "Gender": gender.strip(),
        "Country": country.strip(),
        "Academic_Level": academic_level.strip(),
        "Avg_Daily_Usage_Hours": usage,
        "Most_Used_Platform": platform.strip(),
        "Daily_Unlocks": unlocks,
        "Study_Hours": study,
        "Sleep_Hours_Per_Night": sleep,
        "Physical_Activity_Hours": activity,
        "Stress_Level": stress,
        "Purpose_Of_Use": purpose.strip(),
    }

    try:
        response = requests.post(API_URL, json=student_data, timeout=10)

        if response.status_code == 200:
            result = response.json()
            st.success("Prediction completed!")
            st.metric(
                "Predicted Mental Health Score",
                result["predicted_mental_health_score"]
            )
            st.caption(
                "This is a machine-learning estimate from your project dataset, "
                "not a medical or psychological diagnosis."
            )
        else:
            st.error(f"API error: {response.text}")

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to FastAPI. First run: uvicorn main:app --reload"
        )
    except requests.exceptions.RequestException as error:
        st.error(f"Request failed: {error}")
