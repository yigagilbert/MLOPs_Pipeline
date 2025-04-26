import os
import time
import json
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import mlflow
from prometheus_client import Counter, Gauge, Histogram, Summary

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("monitoring.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define Prometheus metrics
MODEL_REQUESTS = Counter('model_requests_total', 'Total model requests', ['endpoint', 'status'])
MODEL_PREDICTION_LATENCY = Histogram('model_prediction_latency_seconds', 'Model prediction latency in seconds')
MODEL_DRIFT_SCORE = Gauge('model_drift_score', 'Current model drift score')
MODEL_ACCURACY = Gauge('model_accuracy', 'Current model accuracy')
CLASS_DISTRIBUTION = Counter('class_distribution', 'Prediction class distribution', ['class'])

class ModelMonitor:
    """
    Monitoring system for ML model deployment
    """
    def __init__(
        self,
        model_name: str = "resnext",
        log_dir: str = "logs",
        prediction_log_file: str = "predictions.jsonl",
        mlflow_tracking_uri: str = "./mlruns"
    ):
        self.model_name = model_name
        self.log_dir = log_dir
        self.prediction_log_file = os.path.join(log_dir, prediction_log_file)
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup MLflow
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Initialize drift detection
        self.reference_data = None
        self.reference_window_size = 1000  # Number of predictions to use as reference
        self.drift_threshold = 0.1  # Threshold for drift detection
        
        logger.info(f"Model monitor initialized for {model_name}")
    
    def log_prediction(self, prediction_data: Dict[str, Any]):
        """
        Log a prediction to file and update metrics
        
        Args:
            prediction_data: Dictionary with prediction details
        """
        # Add timestamp
        prediction_data["timestamp"] = datetime.now().isoformat()
        
        # Log to file
        with open(self.prediction_log_file, "a") as f:
            f.write(json.dumps(prediction_data) + "\n")
        
        # Update Prometheus metrics
        MODEL_REQUESTS.labels(endpoint="predict", status="success").inc()
        MODEL_PREDICTION_LATENCY.observe(prediction_data.get("inference_time", 0))
        CLASS_DISTRIBUTION.labels(class_=prediction_data.get("prediction", "unknown")).inc()
        
        # Log to MLflow
        self._log_to_mlflow(prediction_data)
    
    def log_error(self, endpoint: str, error: str):
        """
        Log an error
        
        Args:
            endpoint: API endpoint where error occurred
            error: Error message
        """
        MODEL_REQUESTS.labels(endpoint=endpoint, status="error").inc()
        logger.error(f"Error in {endpoint}: {error}")
    
    def _log_to_mlflow(self, prediction_data: Dict[str, Any]):
        """
        Log prediction to MLflow
        
        Args:
            prediction_data: Dictionary with prediction details
        """
        try:
            with mlflow.start_run(run_name=f"{self.model_name}_monitoring", nested=True):
                # Log metrics
                mlflow.log_metric("inference_time", prediction_data.get("inference_time", 0))
                mlflow.log_metric(f"prediction_{prediction_data.get('prediction', 'unknown')}", 1)
                mlflow.log_metric("confidence", prediction_data.get("confidence", 0))
        except Exception as e:
            logger.error(f"Error logging to MLflow: {str(e)}")
    
    def compute_drift(self):
        """
        Compute drift between current and reference data distributions
        """
        try:
            # Load recent predictions
            df = self._load_recent_predictions()
            if df is None or len(df) < 100:  # Need enough data
                logger.warning("Not enough data for drift detection")
                return None
            
            # Initialize reference data if not exists
            if self.reference_data is None:
                earliest_predictions = df.iloc[:self.reference_window_size]
                self.reference_data = {
                    "class_distribution": earliest_predictions["prediction"].value_counts(normalize=True).to_dict(),
                    "confidence_mean": earliest_predictions["confidence"].mean(),
                    "confidence_std": earliest_predictions["confidence"].std()
                }
                logger.info("Reference data initialized")
                return None
            
            # Get recent predictions for comparison
            recent_predictions = df.iloc[-self.reference_window_size:]
            
            # Compute class distribution drift
            recent_distribution = recent_predictions["prediction"].value_counts(normalize=True).to_dict()
            
            # Ensure all classes are represented
            all_classes = set(self.reference_data["class_distribution"].keys()) | set(recent_distribution.keys())
            
            # Calculate JS divergence
            js_divergence = 0
            for cls in all_classes:
                p = self.reference_data["class_distribution"].get(cls, 0)
                q = recent_distribution.get(cls, 0)
                # Add small epsilon to avoid log(0)
                epsilon = 1e-10
                if p > 0 and q > 0:
                    m = (p + q) / 2
                    js_divergence += 0.5 * (p * np.log((p + epsilon) / (m + epsilon)) + 
                                           q * np.log((q + epsilon) / (m + epsilon)))
            
            # Compute confidence statistics drift
            conf_mean_drift = abs(recent_predictions["confidence"].mean() - self.reference_data["confidence_mean"]) / self.reference_data["confidence_mean"]
            
            # Combined drift score (weighted average)
            drift_score = 0.7 * js_divergence + 0.3 * conf_mean_drift
            
            # Update Prometheus metric
            MODEL_DRIFT_SCORE.set(drift_score)
            
            # Log drift metrics to MLflow
            try:
                with mlflow.start_run(run_name=f"{self.model_name}_drift", nested=True):
                    mlflow.log_metric("drift_score", drift_score)
                    mlflow.log_metric("class_distribution_drift", js_divergence)
                    mlflow.log_metric("confidence_drift", conf_mean_drift)
            except Exception as e:
                logger.error(f"Error logging drift to MLflow: {str(e)}")
            
            # Check if drift exceeds threshold
            if drift_score > self.drift_threshold:
                logger.warning(f"Model drift detected! Score: {drift_score:.4f}")
                return drift_score
            
            logger.info(f"Drift score: {drift_score:.4f} (below threshold)")
            return drift_score
            
        except Exception as e:
            logger.error(f"Error computing drift: {str(e)}")
            return None
    
    def _load_recent_predictions(self, days: int = 7) -> Optional[pd.DataFrame]:
        """
        Load recent predictions from log file
        
        Args:
            days: Number of days to look back
            
        Returns:
            DataFrame with recent predictions or None if no data
        """
        try:
            if not os.path.exists(self.prediction_log_file):
                return None
            
            # Read JSONL file
            df = pd.read_json(self.prediction_log_file, lines=True)
            
            if len(df) == 0:
                return None
            
            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # Filter to recent days
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_df = df[df["timestamp"] > cutoff_date]
            
            return recent_df
        
        except Exception as e:
            logger.error(f"Error loading predictions: {str(e)}")
            return None
    
    def update_accuracy(self, ground_truth: List[str], predictions: List[str]):
        """
        Update model accuracy based on ground truth labels
        
        Args:
            ground_truth: List of ground truth labels
            predictions: List of predicted labels
        """
        if len(ground_truth) != len(predictions):
            logger.error("Ground truth and predictions must have same length")
            return
        
        correct = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == pred)
        accuracy = correct / len(ground_truth)
        
        # Update Prometheus metric
        MODEL_ACCURACY.set(accuracy)
        
        # Log to MLflow
        try:
            with mlflow.start_run(run_name=f"{self.model_name}_accuracy", nested=True):
                mlflow.log_metric("accuracy", accuracy)
        except Exception as e:
            logger.error(f"Error logging accuracy to MLflow: {str(e)}")
        
        logger.info(f"Updated model accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    # Example usage
    monitor = ModelMonitor()
    
    # Example prediction data
    prediction = {
        "filename": "test.jpg",
        "prediction": "dementia",
        "confidence": 0.89,
        "inference_time": 0.045
    }
    
    # Log prediction
    monitor.log_prediction(prediction)
    
    # Compute drift (will need more data in real usage)
    drift_score = monitor.compute_drift()
    print(f"Drift score: {drift_score}")