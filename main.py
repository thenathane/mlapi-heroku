import json
import pickle
import joblib
import uvicorn
import nest_asyncio

import numpy as np
import pandas as pd
import xgboost as xgb

from pyngrok import ngrok
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# loading the saved model
model = pickle.load(open('bounce_model.sav', 'rb'))

class model_input(BaseModel):

    Tenure: float
    WarehouseToHome: float
    HourSpendOnApp: float
    NumberOfDeviceRegistered: int
    SatisfactionScore: int
    Complain: int
    OrderAmountHikeFromlastYear: float
    CouponUsed: float
    OrderCount: float
    DaySinceLastOrder: float
    CashbackAmount: int
    PreferredLoginDevice_Mobile_Phone: int
    PreferredLoginDevice_Phone: int
    CityTier_2: int
    CityTier_3: int
    PreferredPaymentMode_COD: int
    PreferredPaymentMode_Cash_on_Delivery: int
    PreferredPaymentMode_Credit_Card: int
    PreferredPaymentMode_Debit_Card: int
    PreferredPaymentMode_E_wallet: int
    PreferredPaymentMode_UPI: int
    Gender_Male: int
    PreferedOrderCat_Grocery: int
    PreferedOrderCat_Laptop_n_Accessory: int
    PreferedOrderCat_Mobile: int
    PreferedOrderCat_Mobile_Phone: int
    PreferedOrderCat_Others: int
    MaritalStatus_Married: int
    MaritalStatus_Single: int
    
@app.post('/predict')
def predict(input_parameters : model_input):

    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)

    try:
        ready = pd.DataFrame([input_dictionary])
        prediction = loaded_model.predict(ready)

        result_list = [int(item) for item in prediction]

        response_data = {
            "success": True,
            "message": "Prediction successful",
            "predictions": result_list
        }

        json_response = json.dumps(response_data)
        return json_response
    except:
        response_data = {
            "success": False,
            "message": "Prediction failed",
            "predictions": []
        }

        return response_data
