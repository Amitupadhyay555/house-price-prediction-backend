import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
import os
import gc
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def load_and_analyze_data():
    """Load and perform initial analysis of the dataset"""
    try:
        # Load the dataset
        df = pd.read_excel('Case_Study_1_Data_1.xlsx')
        
        # Display basic information
        print("\nDataset Shape:", df.shape)
        print("\nDataset Info:")
        print(df.info())
        
        # Display basic statistics
        print("\nBasic Statistics:")
        print(df.describe())
        
        # Check for missing values
        print("\nMissing Values:")
        print(df.isnull().sum())
        
        # Create visualizations
        create_visualizations(df)
        
        return df
    except FileNotFoundError:
        print("Error: 'Case_Study_1_Data_1.xlsx' file not found. Please ensure the file exists in the current directory.")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def create_visualizations(df):
    """Create and save visualizations for data analysis"""
    try:
        # Create directory for plots if it doesn't exist
        if not os.path.exists('plots'):
            os.makedirs('plots')
        
        # Use non-interactive backend to prevent GUI issues
        plt.ioff()
        
        # Price distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(df['Price'].dropna(), bins=50)
        plt.title('Distribution of House Prices')
        plt.xlabel('Price')
        plt.ylabel('Count')
        plt.savefig('plots/price_distribution.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        # Correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        plt.figure(figsize=(12, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.savefig('plots/correlation_heatmap.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        # Box plots for key features
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        if 'Bedrooms' in df.columns:
            sns.boxplot(x='Bedrooms', y='Price', data=df, ax=axes[0,0])
        if 'Bathrooms' in df.columns:
            sns.boxplot(x='Bathrooms', y='Price', data=df, ax=axes[0,1])
        if 'Size' in df.columns:
            sns.scatterplot(x='Size', y='Price', data=df, ax=axes[1,0])
        if 'Condition' in df.columns:
            sns.boxplot(x='Condition', y='Price', data=df, ax=axes[1,1])
        
        plt.tight_layout()
        plt.savefig('plots/feature_analysis.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        # Clear memory
        plt.clf()
        gc.collect()
        
        print("Visualizations saved successfully in 'plots' directory")
        
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")

def preprocess_data(df):
    """Preprocess the data for modeling"""
    try:
        # Handle missing values
        if 'Size' in df.columns:
            df['Size'] = df['Size'].fillna(df['Size'].median())
        if 'Bedrooms' in df.columns:
            df['Bedrooms'] = df['Bedrooms'].fillna(df['Bedrooms'].median())
        if 'Bathrooms' in df.columns:
            df['Bathrooms'] = df['Bathrooms'].fillna(df['Bathrooms'].median())
        if 'Year Built' in df.columns:
            df['Year Built'] = df['Year Built'].fillna(df['Year Built'].median())
        if 'Condition' in df.columns:
            df['Condition'] = df['Condition'].fillna(df['Condition'].mode()[0])
        if 'Price' in df.columns:
            df['Price'] = df['Price'].fillna(df['Price'].median())
        
        # Convert date columns to datetime if exists
        if 'Date Sold' in df.columns:
            df['Date Sold'] = pd.to_datetime(df['Date Sold'], errors='coerce')
            # Extract year and month from Date Sold
            df['Sale Year'] = df['Date Sold'].dt.year
            df['Sale Month'] = df['Date Sold'].dt.month
            # Drop original Date Sold column
            df = df.drop('Date Sold', axis=1)
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in ['Property ID', 'Price']]
        X = df[feature_cols]
        y = df['Price']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define preprocessing steps
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        # Create preprocessing pipeline
        transformers = []
        if numeric_features:
            transformers.append(('num', StandardScaler(), numeric_features))
        if categorical_features:
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features))
        
        preprocessor = ColumnTransformer(transformers=transformers)
        
        print(f"Numeric features: {numeric_features}")
        print(f"Categorical features: {categorical_features}")
        
        return X_train, X_test, y_train, y_test, preprocessor
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return None, None, None, None, None

def train_model(X_train, y_train, preprocessor):
    """Train the model with optimized parameters"""
    try:
        # Create pipeline with preprocessing and model
        # Reduced parameters to prevent memory issues
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(
                n_estimators=50,  # Reduced from 200
                max_depth=10,     # Reduced from 20
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=1,         # Use single core to prevent system overload
                max_features='sqrt'  # Reduce feature complexity
            ))
        ])
        
        print("Training model...")
        # Train the model
        model.fit(X_train, y_train)
        
        # Perform cross-validation with fewer folds
        print("Performing cross-validation...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2')  # Reduced from 5
        print(f"\nCross-validation R2 scores: {cv_scores}")
        print(f"Average CV R2 score: {cv_scores.mean():.4f}")
        
        return model
        
    except Exception as e:
        print(f"Error training model: {str(e)}")
        return None

def evaluate_model(model, X_test, y_test):
    """Evaluate the model performance"""
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print("\nModel Performance:")
        print(f"Root Mean Squared Error: {rmse:,.2f}")
        print(f"R-squared Score: {r2:.4f}")
        
        # Create prediction vs actual plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title('Actual vs Predicted House Prices')
        plt.savefig('plots/prediction_analysis.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        # Clear memory
        gc.collect()
        
        return rmse, r2
        
    except Exception as e:
        print(f"Error evaluating model: {str(e)}")
        return None, None

def save_model(model, filename='house_price_model.joblib'):
    """Save the trained model"""
    try:
        joblib.dump(model, filename, compress=3)  # Add compression to reduce file size
        print(f"\nModel saved as {filename}")
        return True
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        return False

def main():
    """Main function"""
    try:
        # Load and analyze data
        print("Loading and analyzing data...")
        df = load_and_analyze_data()
        
        if df is None:
            print("Failed to load data. Exiting...")
            return
        
        # Preprocess data
        print("\nPreprocessing data...")
        X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
        
        if X_train is None:
            print("Failed to preprocess data. Exiting...")
            return
        
        # Train model
        print("\nTraining model...")
        model = train_model(X_train, y_train, preprocessor)
        
        if model is None:
            print("Failed to train model. Exiting...")
            return
        
        # Evaluate model
        print("\nEvaluating model...")
        rmse, r2 = evaluate_model(model, X_test, y_test)
        
        if rmse is not None and r2 is not None:
            # Save model
            success = save_model(model)
            if success:
                print("\nModel training completed successfully!")
            else:
                print("\nModel training completed but failed to save model.")
        else:
            print("\nModel evaluation failed.")
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
    finally:
        # Clean up memory
        gc.collect()

if __name__ == "__main__":
    main()