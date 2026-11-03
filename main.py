from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import json
from pathlib import Path

app = FastAPI(title="Student Mental Health Prediction API")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Mental_Health_Model.pkl"
COUNTRY_PATH = BASE_DIR / "top_countries.json"

# Load the trained pipeline and country list once when the API starts.
try:
    model = joblib.load(MODEL_PATH)
    with open(COUNTRY_PATH, "r", encoding="utf-8") as file:
        top_countries = json.load(file)
except Exception as error:
    raise RuntimeError(
        "Could not load model files. Keep Mental_Health_Model.pkl and "
        "top_countries.json in the same folder as main.py."
    ) from error


class StudentInput(BaseModel):
    Age: int = Field(ge=10, le=100)
    Gender: str
    Country: str
    Academic_Level: str
    Avg_Daily_Usage_Hours: float = Field(ge=0, le=24)
    Most_Used_Platform: str
    Daily_Unlocks: int = Field(ge=0)
    Study_Hours: float = Field(ge=0, le=24)
    Sleep_Hours_Per_Night: float = Field(ge=0, le=24)
    Physical_Activity_Hours: float = Field(ge=0, le=24)
    Stress_Level: str
    Purpose_Of_Use: str


@app.get("/")
def home():
    return {"message": "Student Mental Health Prediction API is running"}


@app.post("/predict")
def predict_mental_health(student: StudentInput):
    if student.Stress_Level not in ["Low", "Medium", "High", "Very High"]:
        raise HTTPException(
            status_code=400,
            detail="Stress_Level must be Low, Medium, High, or Very High."
        )

    data = student.model_dump()

    # The notebook groups the 10 most common countries into their own labels.
    # Every other country is sent to the model as 'Other'.
    country = data.pop("Country")
    data["Grouped_country"] = country if country in top_countries else "Other"

    # These names and columns match the training notebook.
    feature_columns = [
        "Study_Hours",
        "Age",
        "Avg_Daily_Usage_Hours",
        "Daily_Unlocks",
        "Physical_Activity_Hours",
        "Sleep_Hours_Per_Night",
        "Stress_Level",
        "Gender",
        "Academic_Level",
        "Most_Used_Platform",
        "Purpose_Of_Use",
        "Grouped_country",
    ]

    input_df = pd.DataFrame([data])[feature_columns]

    try:
        prediction = float(model.predict(input_df)[0])
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))

    return {
        "predicted_mental_health_score": round(prediction, 2),
        "message": "Prediction generated successfully"
    }
