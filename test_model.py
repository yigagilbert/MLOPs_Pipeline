import os
import pytest
import torch
import tempfile
import numpy as np
from PIL import Image
from torchvision import transforms
from export_model import create_resnext_model

# Create a mock image for testing
def create_test_image(size=(224, 224)):
    """Create a random test image for testing"""
    img = Image.fromarray(np.random.randint(0, 255, (*size, 3), dtype=np.uint8))
    return img

# Test fixtures
@pytest.fixture
def model():
    """Load the model for testing"""
    try:
        # Try to load from JIT format first
        if os.path.exists("exported_model/model.pt"):
            model = torch.jit.load("exported_model/model.pt", map_location="cpu")
        else:
            # Fall back to loading from state dict
            model = create_resnext_model()
            model.load_state_dict(torch.load("best_resnext.pth", map_location="cpu"))
        model.eval()
        return model
    except Exception as e:
        pytest.skip(f"Model loading failed: {str(e)}")

@pytest.fixture
def transform_fn():
    """Create the transformation pipeline"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

# Tests
def test_model_creation():
    """Test that model can be created"""
    model = create_resnext_model()
    assert model is not None
    assert isinstance(model, torch.nn.Module)
    
    # Check output layer
    assert isinstance(model.fc, torch.nn.Linear)
    assert model.fc.out_features == 2

def test_model_forward(model, transform_fn):
    """Test that model forward pass works"""
    # Skip if model is not available
    if model is None:
        pytest.skip("Model not available")
    
    # Create test image
    img = create_test_image()
    img_tensor = transform_fn(img).unsqueeze(0)
    
    # Run inference
    with torch.no_grad():
        output = model(img_tensor)
    
    # Check output shape and properties
    assert output.shape == (1, 2)  # Batch size of 1, 2 classes
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

def test_model_output_range(model, transform_fn):
    """Test that model outputs are in expected range"""
    # Skip if model is not available
    if model is None:
        pytest.skip("Model not available")
    
    # Create test image
    img = create_test_image()
    img_tensor = transform_fn(img).unsqueeze(0)
    
    # Run inference
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
    
    # Check probability properties
    assert probabilities.shape == (1, 2)
    assert torch.all(probabilities >= 0)
    assert torch.all(probabilities <= 1)
    assert torch.isclose(torch.sum(probabilities, dim=1), torch.tensor([1.0]))

def test_model_batch_inference(model, transform_fn):
    """Test that model can handle batch inference"""
    # Skip if model is not available
    if model is None:
        pytest.skip("Model not available")
    
    # Create a batch of test images
    batch_size = 4
    batch_tensors = []
    
    for _ in range(batch_size):
        img = create_test_image()
        img_tensor = transform_fn(img)
        batch_tensors.append(img_tensor)
    
    batch = torch.stack(batch_tensors)
    
    # Run inference
    with torch.no_grad():
        output = model(batch)
    
    # Check output shape and properties
    assert output.shape == (batch_size, 2)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

def test_model_different_sizes(model, transform_fn):
    """Test that model can handle different image sizes"""
    # Skip if model is not available
    if model is None:
        pytest.skip("Model not available")
    
    # Test different image sizes
    for size in [(160, 120), (300, 400), (224, 224)]:
        img = create_test_image(size=size)
        img_tensor = transform_fn(img).unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            output = model(img_tensor)
        
        # Check output
        assert output.shape == (1, 2)
        assert not torch.isnan(output).any()