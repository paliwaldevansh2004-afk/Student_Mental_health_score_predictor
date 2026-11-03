# Student Mental Health Score Predictor

This project uses the trained Random Forest pipeline from `ML_Project.ipynb`.

## 1. Files required

Keep these files in the same folder:

- `main.py` — FastAPI backend
- `app.py` — Streamlit frontend
- `Mental_Health_Model.pkl` — model saved by the notebook
- `top_countries.json` — country list saved by the notebook
- `requirements.txt`

## 2. One small change in your notebook

Your model uses `Grouped_country`, not the original `Country` column.
The API therefore needs the same top-10 country list used during training.

Immediately after the notebook cell that creates `top_countries`, add and run:

```python
import json

with open("top_countries.json", "w", encoding="utf-8") as file:
    json.dump(top_countries, file)
```

Also make sure the model and this JSON file are in the same folder as `main.py`.

Your notebook currently saves the default Random Forest pipeline using:

```python
joblib.dump(rf_pipeline, "Mental_Health_Model.pkl")
```

That filename matches the API code. If you prefer the tuned model, change it to:

```python
joblib.dump(rf_best_pipeline, "Mental_Health_Model.pkl")
```

Then rerun that cell. The API will load whichever pipeline is saved under that filename.

## 3. Install packages

Open a terminal in this folder and run:

```bash
pip install -r requirements.txt
```

Use the same scikit-learn version that was used to train and save the model.
The requirements file follows the `scikit-learn==1.9.0` version specified in your notebook.

## 4. Run FastAPI

In the terminal:

```bash
uvicorn main:app --reload
```

Open these URLs:

- API home: http://127.0.0.1:8000
- Interactive API testing page: http://127.0.0.1:8000/docs

## 5. Run Streamlit

Open a **second terminal** in the same folder and run:

```bash
streamlit run app.py
```

Streamlit opens the frontend in your browser. Keep the FastAPI terminal running too.

## How the project works

Streamlit form → sends student data with `requests.post()` → FastAPI validates it
with a Pydantic model → country is grouped the same way as in training → saved
scikit-learn pipeline preprocesses the input and predicts the score → FastAPI
returns JSON → Streamlit displays the score.

## Important note

The output is a model prediction based on the dataset. It is not a medical
diagnosis or a validated mental-health assessment.
