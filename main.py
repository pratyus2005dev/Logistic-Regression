from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import os

app = FastAPI(
    title=" Logistic Regression Predictor",
    description="Predicts purchase decision based on Age and Salary",
    version="2.0"
)

# Load model and scaler
with open(os.path.join('model', 'model.pkl'), 'rb') as f:
    scaler, model = pickle.load(f)

# Define input schema
class UserInput(BaseModel):
    Age: int
    EstimatedSalary: float

@app.get("/")
def root():
    return {"message": "FastAPI App is running (Gender removed)."}

@app.post("/predict")
def predict(data: UserInput):
    try:
        # Convert to array and scale
        arr = np.array([[data.Age, data.EstimatedSalary]])
        arr_scaled = scaler.transform(arr)
        prediction = model.predict(arr_scaled)[0]

        return {
            "Your Input": data.dict(),
            "Prediction": "Will Purchase" if prediction == 1 else "Will Not Purchase"
        }
    except Exception as e:
        return {"error": str(e)}
