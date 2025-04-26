import os
import pytest
import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from app import app

# Create test client
client = TestClient(app)

# Helper function to create a test image file in memory
def create_test_image_file(size=(224, 224)):
    """Create a random test image and return as bytes buffer"""
    img = Image.fromarray(np.random.randint(0, 255, (*size, 3), dtype=np.uint8))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

# Tests
def test_health_check():
    """Test health check endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "healthy"

def test_model_info():
    """Test model info endpoint"""
    response = client.get("/model-info")
    assert response.status_code == 200
    assert "name" in response.json()
    assert "num_classes" in response.json()
    assert "class_names" in response.json()
    assert len(response.json()["class_names"]) == 2

def test_prediction_endpoint():
    """Test prediction endpoint with test image"""
    # Create test file
    test_file = create_test_image_file()
    
    # Make request
    files = {"file": ("test_image.png", test_file, "image/png")}
    response = client.post("/predict", files=files)
    
    # Check response
    assert response.status_code == 200
    
    # Check response structure
    data = response.json()
    assert "filename" in data
    assert "prediction" in data
    assert "confidence" in data
    assert "inference_time" in data
    assert "model_info" in data
    
    # Check prediction format
    assert data["prediction"] in ["nodementia", "dementia"]
    assert 0 <= data["confidence"] <= 1
    assert data["inference_time"] > 0

def test_invalid_file_type():
    """Test prediction with invalid file type"""
    # Create text file
    text_file = io.BytesIO(b"This is not an image")
    
    # Make request
    files = {"file": ("test.txt", text_file, "text/plain")}
    response = client.post("/predict", files=files)
    
    # Check response (should be an error)
    assert response.status_code == 400
    assert "detail" in response.json()

def test_no_file():
    """Test prediction with no file"""
    response = client.post("/predict", files={})
    assert response.status_code == 400
    assert "detail" in response.json()

def test_different_image_sizes():
    """Test prediction with different sized images"""
    for size in [(160, 120), (300, 400), (224, 224)]:
        test_file = create_test_image_file(size=size)
        files = {"file": (f"test_{size[0]}x{size[1]}.png", test_file, "image/png")}
        response = client.post("/predict", files=files)
        
        # All should work with resizing
        assert response.status_code == 200
        assert "prediction" in response.json()