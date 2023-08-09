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
    
    tenure = input_dictionary['Tenure']
    warehouseToHome = input_dictionary['WarehouseToHome']
    hourSpendOnApp = input_dictionary['HourSpendOnApp']
    numberOfDeviceRegistered = input_dictionary['NumberOfDeviceRegistered']
    satisfactionScore = input_dictionary['SatisfactionScore']
    complain = input_dictionary['Complain']
    orderAmountHikeFromlastYear = input_dictionary['OrderAmountHikeFromlastYear']
    couponUsed = input_dictionary['CouponUsed']
    orderCount = input_dictionary['OrderCount']
    daySinceLastOrder = input_dictionary['DaySinceLastOrder']
    cashbackAmount = input_dictionary['CashbackAmount']
    preferredLoginDevice_Mobile_Phone = input_dictionary['PreferredLoginDevice_Mobile_Phone']
    preferredLoginDevice_Phone = input_dictionary['PreferredLoginDevice_Phone']
    cityTier_2 = input_dictionary['CityTier_2']
    cityTier_3 = input_dictionary['CityTier_3']
    preferredPaymentMode_COD = input_dictionary['PreferredPaymentMode_COD']
    preferredPaymentMode_Cash_on_Delivery = input_dictionary['PreferredPaymentMode_Cash_on_Delivery']
    preferredPaymentMode_Credit_Card = input_dictionary['PreferredPaymentMode_Credit_Card']
    preferredPaymentMode_Debit_Card = input_dictionary['PreferredPaymentMode_Debit_Card']
    preferredPaymentMode_E_wallet = input_dictionary['PreferredPaymentMode_E_wallet']
    preferredPaymentMode_UPI = input_dictionary['PreferredPaymentMode_UPI']
    gender_Male = input_dictionary['Gender_Male']
    preferedOrderCat_Grocery = input_dictionary['PreferedOrderCat_Grocery']
    preferedOrderCat_Laptop_n_Accessory = input_dictionary['PreferedOrderCat_Laptop_n_Accessory']
    preferedOrderCat_Mobile = input_dictionary['PreferedOrderCat_Mobile']
    preferedOrderCat_Mobile_Phone = input_dictionary['PreferedOrderCat_Mobile_Phone']
    preferedOrderCat_Others = input_dictionary['PreferedOrderCat_Others']
    maritalStatus_Married = input_dictionary['MaritalStatus_Married']
    maritalStatus_Single = input_dictionary['MaritalStatus_Single']

    input_list = [
        tenure,
        warehouseToHome,
        hourSpendOnApp,
        numberOfDeviceRegistered,
        satisfactionScore,
        complain,
        orderAmountHikeFromlastYear,
        couponUsed,
        orderCount,
        daySinceLastOrder,
        cashbackAmount,
        preferredLoginDevice_Mobile_Phone,
        preferredLoginDevice_Phone,
        cityTier_2,
        cityTier_3,
        preferredPaymentMode_COD,
        preferredPaymentMode_Cash_on_Delivery,
        preferredPaymentMode_Credit_Card,
        preferredPaymentMode_Debit_Card,
        preferredPaymentMode_E_wallet,
        preferredPaymentMode_UPI,
        gender_Male,
        preferedOrderCat_Grocery,
        preferedOrderCat_Laptop_n_Accessory,
        preferedOrderCat_Mobile,
        preferedOrderCat_Mobile_Phone,
        preferedOrderCat_Others,
        maritalStatus_Married,
        maritalStatus_Single
    ]
    
    prediction = model.predict(np.array(input_list).reshape(1, -1))
    return prediction

