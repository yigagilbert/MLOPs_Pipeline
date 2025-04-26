import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import logging
from typing import Tuple, List, Dict, Optional
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataValidator:
    """
    Validates input data for model training and inference
    """
    def __init__(
        self,
        min_image_size: Tuple[int, int] = (100, 100),
        max_image_size: Tuple[int, int] = (4000, 4000),
        required_columns: List[str] = ["spectrogram_path", "label"],
        valid_labels: List[str] = ["nodementia", "dementia"]
    ):
        self.min_image_size = min_image_size
        self.max_image_size = max_image_size
        self.required_columns = required_columns
        self.valid_labels = valid_labels
        self.validation_results = {}
    
    def validate_csv(self, csv_path: str) -> Dict:
        """
        Validate CSV file with spectrogram paths and labels
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            Dictionary with validation results
        """
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            return {"valid": False, "error": "File not found"}
        
        try:
            df = pd.read_csv(csv_path)
            results = {
                "total_rows": len(df),
                "missing_columns": [col for col in self.required_columns if col not in df.columns],
                "missing_paths": 0,
                "invalid_labels": 0,
                "valid_samples": 0
            }
            
            # Check required columns
            if results["missing_columns"]:
                logger.error(f"Missing required columns: {results['missing_columns']}")
                return {"valid": False, "error": f"Missing columns: {results['missing_columns']}", **results}
            
            # Check file paths and labels
            valid_samples = []
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Validating {csv_path}"):
                valid_sample = True
                
                # Check if file exists
                if not os.path.exists(row["spectrogram_path"]):
                    results["missing_paths"] += 1
                    valid_sample = False
                
                # Check if label is valid
                if "label" in row and row["label"] not in self.valid_labels:
                    results["invalid_labels"] += 1
                    valid_sample = False
                
                if valid_sample:
                    results["valid_samples"] += 1
                    valid_samples.append(idx)
            
            # Create a filtered dataframe with only valid samples
            filtered_df = df.iloc[valid_samples].copy() if valid_samples else None
            
            validity = (results["valid_samples"] > 0) and (len(results["missing_columns"]) == 0)
            
            return {
                "valid": validity,
                "filtered_df": filtered_df,
                **results
            }
        
        except Exception as e:
            logger.error(f"Error validating CSV: {str(e)}")
            return {"valid": False, "error": str(e)}
    
    def validate_image(self, image_path: str) -> Dict:
        """
        Validate a single image file
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with validation results
        """
        if not os.path.exists(image_path):
            return {"valid": False, "error": "File not found"}
        
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                channels = len(img.getbands())
                
                results = {
                    "size": (width, height),
                    "channels": channels,
                    "format": img.format,
                    "mode": img.mode
                }
                
                # Check size constraints
                if width < self.min_image_size[0] or height < self.min_image_size[1]:
                    return {
                        "valid": False, 
                        "error": f"Image too small. Min size: {self.min_image_size}", 
                        **results
                    }
                
                if width > self.max_image_size[0] or height > self.max_image_size[1]:
                    return {
                        "valid": False, 
                        "error": f"Image too large. Max size: {self.max_image_size}", 
                        **results
                    }
                
                return {"valid": True, **results}
        
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def validate_batch(self, image_paths: List[str]) -> Dict:
        """
        Validate a batch of images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "total": len(image_paths),
            "valid": 0,
            "invalid": 0,
            "errors": {}
        }
        
        for path in tqdm(image_paths, desc="Validating images"):
            img_result = self.validate_image(path)
            if img_result["valid"]:
                results["valid"] += 1
            else:
                results["invalid"] += 1
                results["errors"][path] = img_result["error"]
        
        return results
    
    def export_validation_report(self, output_path: str = "validation_report.yaml"):
        """
        Export validation results to a YAML file
        
        Args:
            output_path: Path to output YAML file
        """
        with open(output_path, "w") as f:
            yaml.dump(self.validation_results, f)
        
        logger.info(f"Validation report saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    validator = DataValidator()
    
    # Validate training data
    train_results = validator.validate_csv("train_spectrogram_tracking.csv")
    if train_results["valid"]:
        print(f"Training data valid: {train_results['valid_samples']} valid samples")
        # Optionally validate a sample of images
        if train_results["filtered_df"] is not None:
            sample_paths = train_results["filtered_df"]["spectrogram_path"].sample(min(10, len(train_results["filtered_df"]))).tolist()
            image_results = validator.validate_batch(sample_paths)
            print(f"Image validation: {image_results['valid']} valid, {image_results['invalid']} invalid")
    else:
        print(f"Training data invalid: {train_results.get('error', 'Unknown error')}")
    
    # Store validation results
    validator.validation_results["train"] = train_results
    
    # Export report
    validator.export_validation_report()