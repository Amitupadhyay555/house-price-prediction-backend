import gc
import os
import json
import logging
from datetime import datetime
from typing import Optional, List
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import joblib
import pandas as pd
import uvicorn
from property_matching import PropertyMatcher
import numpy as np

# Configure logging
logging.basicConfig(
    filename='api.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Property Matching API",
    description="API for matching properties with user preferences",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
try:
    model = joblib.load('house_price_model.joblib')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# Initialize property matcher
property_matcher = PropertyMatcher()
try:
    property_matcher.load_data()
    logger.info("Property matching data loaded successfully")
except Exception as e:
    logger.error(f"Error loading property matching data: {str(e)}")

# Load prediction history
prediction_history = []
HISTORY_FILE = 'prediction_history.json'

def load_prediction_history():
    """Load prediction history from file"""
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        logger.error(f"Error loading prediction history: {str(e)}")
        return []

def save_prediction_history(history):
    """Save prediction history to file"""
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving prediction history: {str(e)}")

# Load initial history
prediction_history = load_prediction_history()
logger.info(f"Loaded {len(prediction_history)} predictions from history")

# Models for property matching
class UserPreference(BaseModel):
    user_id: int
    budget_min: float
    budget_max: float
    preferred_locations: list
    min_bedrooms: int
    min_bathrooms: int
    preferred_property_type: str

class PropertyMatch(BaseModel):
    property_id: int
    price: float
    location: str
    description: str
    match_score: float
    budget_score: float
    location_score: float
    property_type_score: float
    features_score: float
    description_score: float

class PropertyFeatures(BaseModel):
    Size: float = Field(..., gt=0, description="Size in square feet")
    Year_Built: int = Field(..., gt=1800, description="Year the property was built")
    Bedrooms: int = Field(..., gt=0, description="Number of bedrooms")
    Bathrooms: int = Field(..., gt=0, description="Number of bathrooms")
    Stories: int = Field(..., gt=0, description="Number of stories")
    Parking: int = Field(..., ge=0, description="Number of parking spaces")
    Location: str = Field(..., description="Property location")
    Condition: str = Field(..., description="Property condition")
    Type: str = Field(..., description="Property type")
    Sale_Year: int = Field(..., description="Year of sale")
    Sale_Month: int = Field(..., ge=1, le=12, description="Month of sale")

class PredictionResponse(BaseModel):
    predicted_price: float
    input_features: PropertyFeatures

class MatchRequest(BaseModel):
    user_id: int
    top_n: int = 5

# Property matching endpoints
@app.post("/match-properties", response_model=List[PropertyMatch])
async def match_properties(request: MatchRequest):
    """Get top property matches for a user"""
    try:
        matches = property_matcher.get_top_matches(request.user_id, request.top_n)
        return matches
    except Exception as e:
        logger.error(f"Error matching properties: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/property-match/{user_id}/{property_id}")
async def get_property_match(user_id: int, property_id: int):
    """Get match score for a specific user-property pair"""
    try:
        matches = property_matcher.get_top_matches(user_id)
        match = next((m for m in matches if m['property_id'] == property_id), None)
        if match is None:
            raise HTTPException(status_code=404, detail="Property match not found")
        return match
    except Exception as e:
        logger.error(f"Property match error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Keep existing endpoints
@app.get("/")
def read_root():
    """Welcome endpoint"""
    return {
        "message": "Welcome to the Property Matching API",
        "version": "1.0.0",
        "endpoints": {
            "/docs": "API documentation",
            "/match-properties": "Get property matches",
            "/property-match/{user_id}/{property_id}": "Get specific match score",
            "/health": "Health check"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_price(features: PropertyFeatures):
    try:
        # Create a DataFrame with all features
        input_df = pd.DataFrame([{
            'Size': features.Size,
            'Year Built': features.Year_Built,
            'Bedrooms': features.Bedrooms,
            'Bathrooms': features.Bathrooms,
            'Stories': features.Stories,
            'Parking': features.Parking,
            'Sale Year': features.Sale_Year,
            'Sale Month': features.Sale_Month,
            'Condition': features.Condition,
            'Type': features.Type,
            'Location': features.Location
        }])
        
        # Make prediction using the pipeline
        predicted_price = float(model.predict(input_df)[0])
        
        # Create prediction record
        prediction = {
            "timestamp": datetime.now().isoformat(),
            "input": {
                "Size": features.Size,
                "Year_Built": features.Year_Built,
                "Bedrooms": features.Bedrooms,
                "Bathrooms": features.Bathrooms,
                "Stories": features.Stories,
                "Parking": features.Parking,
                "Location": features.Location,
                "Condition": features.Condition,
                "Type": features.Type,
                "Sale_Year": features.Sale_Year,
                "Sale_Month": features.Sale_Month
            },
            "predicted_price": predicted_price
        }
        
        # Add to history
        prediction_history.append(prediction)
        
        # Keep only last 100 predictions
        if len(prediction_history) > 100:
            prediction_history.pop(0)
        
        # Save updated history
        save_prediction_history(prediction_history)
        
        return PredictionResponse(
            predicted_price=predicted_price,
            input_features=features
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def get_prediction_history():
    """Get prediction history"""
    try:
        # Reload history from file to ensure we have the latest data
        history = load_prediction_history()
        return history
    except Exception as e:
        logger.error(f"Error getting prediction history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     logger.info("Starting API server...")
#     uvicorn.run(
#         app,
#         host="127.0.0.1",
#         port=8000,
#         workers=1,
#         limit_concurrency=10,
#         timeout_keep_alive=5,
#         access_log=False,
#         log_level="info"
#     )