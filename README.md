# House Price Prediction Model

This project implements a machine learning model to predict house prices based on various property features. The solution includes comprehensive data analysis, model training, and a production-ready API with multi-core processing support.

## Project Structure

- `house_price_analysis.py`: Main script for data analysis and model training
- `api.py`: FastAPI implementation for serving the model
- `requirements.txt`: Project dependencies
- `house_price_model.joblib`: Trained model file (generated after running the analysis)
- `plots/`: Directory containing data visualizations
- `prediction_history.json`: History of predictions made through the API
- `api.log`: API server logs

## Features

### Data Analysis and Visualization
- Comprehensive data preprocessing
- Missing value handling
- Feature correlation analysis
- Distribution analysis
- Interactive visualizations saved as plots

### Model Development
- Random Forest Regressor with optimized hyperparameters
- Cross-validation for robust performance evaluation
- Feature importance analysis
- Model performance metrics (RMSE, R²)
- Prediction vs actual visualization

### API Features
- RESTful API with FastAPI
- Input validation and error handling
- Multi-core processing support
- Prediction history tracking
- Health check endpoint
- CORS support
- API documentation (Swagger UI and ReDoc)
- Logging system

## Setup and Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Train the model and generate visualizations:
```bash
python house_price_analysis.py
```

2. Start the API server:
```bash
python api.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

- `GET /`: Welcome message and API information
- `POST /predict`: Predict house price based on property features
- `GET /history`: View prediction history
- `GET /health`: Health check endpoint
- `GET /docs`: Interactive API documentation (Swagger UI)
- `GET /redoc`: Alternative API documentation (ReDoc)

### Example API Request

```python
import requests

data = {
    "Location": "Seattle",
    "Size": 2000,
    "Bedrooms": 3,
    "Bathrooms": 2.5,
    "Year_Built": 2010,
    "Condition": "Good",
    "Type": "Single Family",
    "Sale_Year": 2023,
    "Sale_Month": 6
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

## Model Performance

The model's performance is evaluated using:
- Root Mean Squared Error (RMSE)
- R-squared Score
- Cross-validation scores
- Prediction vs actual plots

## Multi-core Processing

- Model training uses all available CPU cores
- API server automatically scales based on available CPU cores
- Background tasks for non-blocking operations

## Data Validation

The API includes comprehensive input validation:
- Numeric range checks
- Date validation
- Required field validation
- Data type validation

## Logging and Monitoring

- Detailed API logs in `api.log`
- Prediction history tracking
- Health check endpoint for monitoring
- Error tracking and reporting

## Notes

- The model uses Random Forest Regressor with optimized parameters
- Multi-core processing is enabled by default
- The API automatically scales based on available CPU cores
- All predictions are logged and can be accessed through the history endpoint 