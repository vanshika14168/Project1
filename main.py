from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import pandas as pd
import joblib

app = FastAPI(
    title="Housing Price Prediction",
    version="1.0"
)

# Load trained model
model = joblib.load("models/house_model.pkl")

# Configure templates
templates = Jinja2Templates(directory="templates")

# Configure static files
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# Home page
@app.get("/")
def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "prediction": None,
            "error": None
        }
    )


# Prediction endpoint
@app.post("/predict")
def predict(
    request: Request,

    area: float = Form(...),
    bedrooms: int = Form(...),
    bathrooms: int = Form(...),
    stories: int = Form(...),
    parking: int = Form(...),

    mainroad: str = Form(...),
    guestroom: str = Form(...),
    basement: str = Form(...),
    hotwaterheating: str = Form(...),
    airconditioning: str = Form(...),
    prefarea: str = Form(...),

    furnishingstatus: str = Form(...)
):

    input_data = pd.DataFrame([{
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "parking": parking,
        "mainroad": mainroad,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning,
        "prefarea": prefarea,
        "furnishingstatus": furnishingstatus
    }])

    prediction = model.predict(input_data)[0]

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "prediction": round(float(prediction), 2),
            "error": None
        }
    )
