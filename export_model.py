import torch
import torch.nn as nn
from torchvision import models
import os
import mlflow
import yaml

def create_resnext_model():
    """Create the ResNeXt model architecture with the correct classifier"""
    model = models.resnext50_32x4d(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model

def export_model(model_path, output_dir="exported_model"):
    """
    Load model weights and export to various formats for deployment
    
    Args:
        model_path: Path to the trained model weights (.pth file)
        output_dir: Directory to save exported models
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model architecture
    model = create_resnext_model()
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    # Save model architecture info
    model_info = {
        "name": "resnext50_32x4d",
        "num_classes": 2,
        "input_size": [3, 224, 224],
        "class_names": ["nodementia", "dementia"]
    }
    
    with open(os.path.join(output_dir, "model_info.yaml"), "w") as f:
        yaml.dump(model_info, f)
    
    # Export to TorchScript (JIT) for efficient C++ deployment
    example_input = torch.randn(1, 3, 224, 224)
    traced_model = torch.jit.trace(model, example_input)
    torch.jit.save(traced_model, os.path.join(output_dir, "model.pt"))
    
    # Export to ONNX for cross-platform deployment
    torch.onnx.export(
        model,
        example_input,
        os.path.join(output_dir, "model.onnx"),
        export_params=True,
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    
    # Log model to MLflow
    mlflow.set_tracking_uri("./mlruns")
    with mlflow.start_run(run_name="resnext_production_model"):
        mlflow.log_param("model_type", "resnext50_32x4d")
        mlflow.log_param("input_shape", "[3, 224, 224]")
        mlflow.log_param("classes", "nodementia, dementia")
        
        # Log the PyTorch model
        mlflow.pytorch.log_model(model, "pytorch_model")
        
        # Log the ONNX model as an artifact
        mlflow.log_artifact(os.path.join(output_dir, "model.onnx"), "onnx_model")
        
    print(f"Model exported successfully to {output_dir}")
    print("Formats: PyTorch (.pt), ONNX (.onnx), MLflow")

if __name__ == "__main__":
    model_path = "best_resnext.pth"
    export_model(model_path)