from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import json


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    

# loading the saved model
model = pickle.load(open('bounce_model.sav', 'rb'))


@app.post('/predict')
def pred(input_parameters: model_input):
    
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    prediction = model.predict(pd.DataFrame([input_dictionary]))
    return prediction

