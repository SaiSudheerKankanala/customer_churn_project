
from scr.predict import prediction
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal

app = FastAPI()

predict_model = prediction()


class Customers(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int
    PhoneService: Literal["Yes", "No"]
    MultipleLines: str
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
    MonthlyCharges: float
    TotalCharges: float


@app.post("/predict")
def predict(data: Customers):
    customer_data = [{
        "gender": data.gender,
        "SeniorCitizen": data.SeniorCitizen,
        "Partner": data.Partner,
        "Dependents": data.Dependents,
        "tenure": data.tenure,
        "PhoneService": data.PhoneService,
        "MultipleLines": data.MultipleLines,
        "InternetService": data.InternetService,
        "OnlineSecurity": data.OnlineSecurity,
        "OnlineBackup": data.OnlineBackup,
        "DeviceProtection": data.DeviceProtection,
        "TechSupport": data.TechSupport,
        "StreamingTV": data.StreamingTV,
        "StreamingMovies": data.StreamingMovies,
        "Contract": data.Contract,
        "PaperlessBilling": data.PaperlessBilling,
        "PaymentMethod": data.PaymentMethod,
        "MonthlyCharges": data.MonthlyCharges,
        "TotalCharges": data.TotalCharges
    }]

    result = predict_model.predict(customer_data)

    return {"prediction": int(result[0])}