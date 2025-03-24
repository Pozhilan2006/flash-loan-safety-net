import json
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Try to import required libraries
try:
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    import joblib

    # Visualization libraries - optional
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        visualization_available = True
    except ImportError:
        logger.warning("Visualization libraries (matplotlib/seaborn) not found. Visualizations will be skipped.")
        visualization_available = False

except ImportError as e:
    logger.error(f"Required library not found: {str(e)}")
    logger.error("Please install required libraries using: pip install pandas numpy scikit-learn matplotlib seaborn joblib")
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# File Paths
BASE_DIR = Path(__file__).parent.parent
DATA_FILE = BASE_DIR / "flash_loan_data.json"
MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = MODEL_DIR / "flash_loan_risk_model.pkl"
METRICS_FILE = MODEL_DIR / "model_metrics.json"
PLOTS_DIR = BASE_DIR / "plots"

# Ensure directories exist
MODEL_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

def load_data():
    """Load and validate data from JSON file"""
    if not DATA_FILE.exists():
        logger.error(f"Data file not found: {DATA_FILE}")
        return None

    try:
        # Check if file is empty
        if os.path.getsize(DATA_FILE) == 0:
            logger.error(f"Data file is empty: {DATA_FILE}")
            create_sample_data()
            logger.info(f"Created sample data in {DATA_FILE}")

        with open(DATA_FILE, "r") as f:
            file_content = f.read()

        # Check if content is valid JSON
        try:
            data = json.loads(file_content)
        except json.JSONDecodeError:
            logger.error("Invalid JSON format in data file. Attempting to fix...")
            create_sample_data()
            with open(DATA_FILE, "r") as f:
                data = json.load(f)

        reserves = data.get("data", {}).get("reserves", [])

        if not reserves:
            logger.error("No reserves data found in JSON file. Creating sample data...")
            create_sample_data()
            with open(DATA_FILE, "r") as f:
                data = json.load(f)
            reserves = data.get("data", {}).get("reserves", [])

        logger.info(f"Successfully loaded data with {len(reserves)} records")
        return reserves
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return None

def create_sample_data():
    """Create sample data if the JSON file is empty or corrupted"""
    logger.info("Creating sample data file with valid format")

    # Create a larger sample data structure with more variety
    reserves = []

    # Low risk assets (variable rate < 0.05)
    low_risk_assets = [
        {
            "symbol": "DAI",
            "liquidityRate": "0.03",
            "variableBorrowRate": "0.04",
            "stableBorrowRate": "0.05",
            "totalLiquidity": "1000000"
        },
        {
            "symbol": "USDC",
            "liquidityRate": "0.025",
            "variableBorrowRate": "0.035",
            "stableBorrowRate": "0.045",
            "totalLiquidity": "2000000"
        },
        {
            "symbol": "USDT",
            "liquidityRate": "0.02",
            "variableBorrowRate": "0.03",
            "stableBorrowRate": "0.04",
            "totalLiquidity": "1500000"
        },
        {
            "symbol": "ETH",
            "liquidityRate": "0.01",
            "variableBorrowRate": "0.02",
            "stableBorrowRate": "0.03",
            "totalLiquidity": "500000"
        },
        {
            "symbol": "WBTC",
            "liquidityRate": "0.015",
            "variableBorrowRate": "0.025",
            "stableBorrowRate": "0.035",
            "totalLiquidity": "300000"
        },
        {
            "symbol": "BUSD",
            "liquidityRate": "0.022",
            "variableBorrowRate": "0.032",
            "stableBorrowRate": "0.042",
            "totalLiquidity": "800000"
        },
        {
            "symbol": "TUSD",
            "liquidityRate": "0.024",
            "variableBorrowRate": "0.034",
            "stableBorrowRate": "0.044",
            "totalLiquidity": "600000"
        }
    ]

    # High risk assets (variable rate > 0.05)
    high_risk_assets = [
        {
            "symbol": "LINK",
            "liquidityRate": "0.02",
            "variableBorrowRate": "0.06",
            "stableBorrowRate": "0.07",
            "totalLiquidity": "150000"
        },
        {
            "symbol": "UNI",
            "liquidityRate": "0.025",
            "variableBorrowRate": "0.07",
            "stableBorrowRate": "0.08",
            "totalLiquidity": "100000"
        },
        {
            "symbol": "AAVE",
            "liquidityRate": "0.03",
            "variableBorrowRate": "0.08",
            "stableBorrowRate": "0.09",
            "totalLiquidity": "80000"
        },
        {
            "symbol": "SNX",
            "liquidityRate": "0.015",
            "variableBorrowRate": "0.09",
            "stableBorrowRate": "0.10",
            "totalLiquidity": "50000"
        },
        {
            "symbol": "MKR",
            "liquidityRate": "0.02",
            "variableBorrowRate": "0.065",
            "stableBorrowRate": "0.075",
            "totalLiquidity": "120000"
        },
        {
            "symbol": "COMP",
            "liquidityRate": "0.025",
            "variableBorrowRate": "0.075",
            "stableBorrowRate": "0.085",
            "totalLiquidity": "90000"
        },
        {
            "symbol": "YFI",
            "liquidityRate": "0.01",
            "variableBorrowRate": "0.085",
            "stableBorrowRate": "0.095",
            "totalLiquidity": "40000"
        },
        {
            "symbol": "SUSHI",
            "liquidityRate": "0.015",
            "variableBorrowRate": "0.095",
            "stableBorrowRate": "0.105",
            "totalLiquidity": "30000"
        }
    ]

    # Add all assets to the reserves
    reserves.extend(low_risk_assets)
    reserves.extend(high_risk_assets)

    # Create the final data structure
    sample_data = {
        "data": {
            "reserves": reserves
        }
    }

    # Write the sample data to the file
    try:
        # Ensure the directory exists
        DATA_FILE.parent.mkdir(exist_ok=True)

        with open(DATA_FILE, "w") as f:
            json.dump(sample_data, f, indent=4)

        logger.info(f"Sample data written to {DATA_FILE}")
    except Exception as e:
        logger.error(f"Error creating sample data: {str(e)}")

    return True

def preprocess_data(data):
    """Convert data to DataFrame and engineer features"""
    if not data:
        logger.error("No valid data provided for preprocessing")
        return None

    try:
        df = pd.DataFrame(data)

        # Validate required columns
        required_columns = ["liquidityRate", "variableBorrowRate", "stableBorrowRate"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return None

        # Convert string values to float
        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows with NaN values
        initial_rows = len(df)
        df = df.dropna(subset=required_columns)
        if len(df) < initial_rows:
            logger.warning(f"Dropped {initial_rows - len(df)} rows with missing values")

        # Feature engineering
        df["rate_spread"] = df["variableBorrowRate"] - df["liquidityRate"]
        df["rate_ratio"] = df["variableBorrowRate"] / (df["liquidityRate"] + 1e-10)  # Avoid division by zero
        df["stable_variable_diff"] = df["stableBorrowRate"] - df["variableBorrowRate"]

        # Create more sophisticated risk label
        # Higher risk when variable rate is high AND spread between rates is large
        df["risk_score"] = (
            (df["variableBorrowRate"] > 0.05).astype(int) +
            (df["rate_spread"] > 0.03).astype(int) +
            (df["rate_ratio"] > 2).astype(int)
        )

        # Binary classification: high risk (2-3 points) vs low risk (0-1 points)
        df["risk_label"] = (df["risk_score"] >= 2).astype(int)

        logger.info(f"Preprocessing complete. DataFrame shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        return None

def visualize_data(df):
    """Create visualizations of the data"""
    if not visualization_available:
        logger.warning("Skipping visualizations due to missing libraries")
        return

    try:
        # Distribution of risk labels
        plt.figure(figsize=(10, 6))
        sns.countplot(x='risk_label', data=df)
        plt.title('Distribution of Risk Labels')
        plt.savefig(PLOTS_DIR / 'risk_distribution.png')

        # Feature correlations
        plt.figure(figsize=(12, 10))
        feature_cols = ["liquidityRate", "variableBorrowRate", "stableBorrowRate",
                        "rate_spread", "rate_ratio", "stable_variable_diff", "risk_label"]
        sns.heatmap(df[feature_cols].corr(), annot=True, cmap='coolwarm')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'feature_correlations.png')

        logger.info(f"Data visualizations saved to {PLOTS_DIR}")
    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")
        logger.error("You may need to install matplotlib and seaborn: pip install matplotlib seaborn")

def train_model(df, tune_hyperparams=False):
    """Train and evaluate the model"""
    if df is None or df.empty:
        logger.error("DataFrame is empty, cannot train model")
        return

    try:
        # Select features
        features = [
            "liquidityRate", "variableBorrowRate", "stableBorrowRate",
            "rate_spread", "rate_ratio", "stable_variable_diff"
        ]
        X = df[features]
        y = df["risk_label"]

        # Check if we have enough data for stratified split
        # We need at least 2 samples of each class for train and test
        min_samples_per_class = y.value_counts().min()

        if len(df) < 10 or min_samples_per_class < 4:
            logger.warning("Small dataset detected. Using simple train/test split without stratification.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
        else:
            # Use stratified split for larger datasets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

        logger.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

        # Train model with hyperparameter tuning if requested
        if tune_hyperparams:
            logger.info("Performing hyperparameter tuning...")
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

            grid_search = GridSearchCV(
                RandomForestClassifier(random_state=42),
                param_grid=param_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1
            )

            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            logger.info(f"Best parameters: {grid_search.best_params_}")
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "feature_importance": dict(zip(features, model.feature_importances_.tolist()))
        }

        # Log metrics
        logger.info(f"Model Performance:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")

        # Save visualization plots if libraries are available
        if visualization_available:
            # Save confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.savefig(PLOTS_DIR / 'confusion_matrix.png')

            # Feature importance plot
            plt.figure(figsize=(10, 6))
            importance = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)

            sns.barplot(x='Importance', y='Feature', data=importance)
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / 'feature_importance.png')

            logger.info(f"Visualization plots saved to {PLOTS_DIR}")

        # Save model and metrics
        joblib.dump(model, MODEL_FILE)
        with open(METRICS_FILE, 'w') as f:
            json.dump(metrics, f, indent=4)

        logger.info(f"Model trained and saved to {MODEL_FILE}")
        logger.info(f"Model metrics saved to {METRICS_FILE}")

        return model, metrics

    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        logger.error("Make sure you have scikit-learn installed: pip install scikit-learn")
        return None, None

def verify_json_file():
    """Verify that the JSON file exists and is properly formatted"""
    logger.info(f"Verifying JSON file: {DATA_FILE}")

    # Check if file exists
    if not DATA_FILE.exists():
        logger.warning(f"JSON file does not exist. Creating sample data file.")
        create_sample_data()
        return True

    # Check if file is empty
    if os.path.getsize(DATA_FILE) == 0:
        logger.warning(f"JSON file is empty. Creating sample data.")
        create_sample_data()
        return True

    # Verify JSON formatting
    try:
        with open(DATA_FILE, "r") as f:
            json.load(f)
        logger.info("JSON file is valid.")
        return True
    except json.JSONDecodeError as e:
        logger.error(f"JSON file is corrupted: {str(e)}")
        logger.info("Creating new sample data file.")
        create_sample_data()
        return True
    except Exception as e:
        logger.error(f"Error verifying JSON file: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting model training process")

    # Force recreation of sample data for this run
    logger.info("Forcing recreation of sample data with expanded dataset")
    create_sample_data()

    # Then load the data
    data = load_data()

    if data:
        df = preprocess_data(data)
        if df is not None:
            visualize_data(df)
            train_model(df, tune_hyperparams=False)
    else:
        logger.error("Failed to load data. Please check the JSON file manually.")

    logger.info("Model training process completed")
