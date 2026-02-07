from custom_transformers import CombinedAttributesAdder
from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import HTMLResponse

from pydantic import BaseModel #pydantic validates input data
#coverts types automatically,  throws errors when data is wrong
#Fast Api respone ...422 unprocessable Entity

#it ensures type sacfety and consistency between tarining and inference ,WHILE 
#TRANSFORMATIOS ARE HANDLED INSIDE A PIPELINE

import joblib

import numpy as np
import pandas as pd

app = FastAPI()

model = joblib.load("housing_model.pkl")

housing_pipeline = joblib.load("housing_pipeline.pkl")

#

class HouseInput(BaseModel):
    median_income: float
    total_bedrooms:float
    total_rooms: float
    population: float
    households:float
    housing_median_age:float
    longitude: float
    latitude: float
    ocean_proximity:str


@app.post("/predict")

def predict(data:HouseInput):

    X = pd.DataFrame([{
        "median_income":data.median_income,
        "total_rooms":data.total_rooms,
        "total_bedrooms":data.total_bedrooms,
        "housing_median_age":data.housing_median_age,
       
        "population":data.population,
        "households":data.households,
        "longitude":data.longitude,
        "latitude":data.latitude,
        "ocean_proximity":data.ocean_proximity
    

    }])

    features = housing_pipeline.transform(X)

    prediction = model.predict(features)[0]

    return {"predicted_price": float(prediction)}

@app.get("/",response_class=HTMLResponse)
def home():
    with open("frontend.html","r" , encoding= "utf-8") as f:

        return f.read()



app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials = True,

    allow_methods=["*"],

    allow_headers=["*"]
)
