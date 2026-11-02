from pathlib import Path
import logging
import os


project_path = Path(__file__).resolve().parent.parent
DataBaseUrl = project_path/"data"/"customer_churn_data.csv"

os.makedirs('logs',exist_ok = True)

logging.basicConfig(level = logging.INFO,
                    format = "%(asctime)s/ %(levelname)s/ %(message)s",
                    handlers = ([logging.FileHandler("logs/project.logs"),
                    logging.StreamHandler()])
)
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URL = "file:./mlrun"
MLFLOW_EXPERIMENT_NAME = "customer_churn_prediction"
MLFLOW_ARTIFACTS_LOCATION = "artifacts"

Target_column = "Churn"
unnecessary_columns = ['customerID']

test_size = 0.2
randomstate = 45


rf_param = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2"],
    "class_weight": [None, "balanced"],
}

xgb_param = {
    "n_estimators": [100, 200, 300, 500],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7, 10],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "gamma": [0, 0.1, 0.3, 0.5],
    "reg_alpha": [0, 0.01, 0.1, 1],
    "reg_lambda": [1, 5, 10],
    "scale_pos_weight": [1, 2, 5, 10]
}

thresholds = [0.2,0.3,0.4,0.5,0.6,0.7,0.8]

min_recall = 0.85
max_recall = 1

artifacts_dir = project_path/"artifacts"
artifacts_dir.mkdir(parents = True, exist_ok=True)
