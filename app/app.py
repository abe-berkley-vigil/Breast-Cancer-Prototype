import pickle

from fastapi import FastAPI
import numpy as np

app = FastAPI()

# Load the trained model
lr_model = pickle.load("../models/model.pkl")


@app.get("/")
def read_root():
    return {"message": "Welcome to Breast Cancer Classification - SE 489 Model"}


@app.post("/predict/")
def predict(data: dict):
    X_test = np.array(data["features"])
    y_pred = lr_model.predict(X_test)
    return {"predictions": y_pred}
