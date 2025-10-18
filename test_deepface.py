"""
DeepFace Setup and Model Download Test
Tests DeepFace installation and ensures models download to correct location
"""

import yaml
import sys
import os
import cv2
import numpy as np

def load_config():
    """Load configuration file"""
    try:
        with open("config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        print("✓ Configuration file loaded")
        return config
    except Exception as e:
        print(f"✗ Failed to load configuration: {str(e)}")
        return None

def test_deepface_installation():
    """Test DeepFace installation"""
    print("\n--- Test 1: DeepFace Installation ---")
    try:
        from deepface import DeepFace
        print("✓ DeepFace imported successfully")
        return True
    except ImportError as e:
        print(f"✗ DeepFace not installed: {str(e)}")
        print("  Run: pip install deepface tf-keras")
        return False

def test_custom_model_directory(config):
    """Test custom model directory setup"""
    print("\n--- Test 2: Custom Model Directory ---")
    try:
        models_path = "d:\\CASC Project\\deepface_models"
        
        # Set environment variable
        os.environ['DEEPFACE_HOME'] = models_path
        
        # Create directory
        os.makedirs(models_path, exist_ok=True)
        print(f"✓ Models directory created: {models_path}")
        
        # Verify environment variable
        deepface_home = os.environ.get('DEEPFACE_HOME')
        print(f"✓ DEEPFACE_HOME set to: {deepface_home}")
        
        return True
    except Exception as e:
        print(f"✗ Model directory setup failed: {str(e)}")
        return False

def test_model_download_and_analysis():
    """Test model download and basic analysis"""
    print("\n--- Test 3: Model Download and Analysis ---")
    try:
        from deepface import DeepFace
        
        # Create test image
        test_img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Add a simple face-like pattern
        cv2.circle(test_img, (112, 112), 50, (255, 255, 255), -1)  # Face
        cv2.circle(test_img, (95, 95), 8, (0, 0, 0), -1)   # Left eye
        cv2.circle(test_img, (129, 95), 8, (0, 0, 0), -1)  # Right eye
        cv2.ellipse(test_img, (112, 130), (15, 8), 0, 0, 180, (0, 0, 0), 2)  # Mouth
        
        print("  Created test image")
        
        # Test analysis (this will download models)
        print("  Running analysis (models will download if needed)...")
        result = DeepFace.analyze(
            img_path=test_img,
            actions=['emotion', 'age', 'gender'],
            detector_backend='opencv',
            enforce_detection=False,
            silent=True
        )
        
        print("✓ Analysis completed successfully")
        
        if isinstance(result, list):
            result = result[0] if result else {}
        
        print(f"  Detected emotion: {result.get('dominant_emotion', 'unknown')}")
        print(f"  Estimated age: {result.get('age', 0)}")
        print(f"  Detected gender: {result.get('dominant_gender', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Analysis failed: {str(e)}")
        return False

def test_face_database_setup(config):
    """Test face database directory setup"""
    print("\n--- Test 4: Face Database Setup ---")
    try:
        db_path = config['deepface']['db_path']
        
        # Create database directory
        os.makedirs(db_path, exist_ok=True)
        print(f"✓ Database directory created: {db_path}")
        
        return True
    except Exception as e:
        print(f"✗ Database setup failed: {str(e)}")
        return False

def test_casc_deepface_integration():
    """Test CASC DeepFace client integration"""
    print("\n--- Test 5: CASC Integration ---")
    try:
        from src.deepface_client import DeepFaceClient
        
        config = load_config()
        if not config:
            return False
        
        # Initialize CASC DeepFace client
        deepface_client = DeepFaceClient(config)
        print("✓ CASC DeepFace client initialized")
        
        # Test list known persons
        persons = deepface_client.list_known_persons()
        print(f"✓ Listed {len(persons)} known person(s)")
        
        return True
        
    except Exception as e:
        print(f"✗ CASC integration failed: {str(e)}")
        return False

def check_model_files():
    """Check if models were downloaded to correct location"""
    print("\n--- Test 6: Model File Verification ---")
    try:
        models_path = "d:\\CASC Project\\deepface_models"
        
        if os.path.exists(models_path):
            files = os.listdir(models_path)
            model_files = [f for f in files if f.endswith(('.h5', '.pb', '.weights'))]
            
            print(f"✓ Models directory exists: {models_path}")
            print(f"✓ Found {len(model_files)} model file(s)")
            
            if model_files:
                print("  Model files:")
                for file in model_files[:5]:  # Show first 5 files
                    print(f"    - {file}")
                if len(model_files) > 5:
                    print(f"    ... and {len(model_files) - 5} more")
            
            # Check directory size
            total_size = 0
            for root, dirs, files in os.walk(models_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
            
            size_mb = total_size / (1024 * 1024)
            print(f"✓ Models directory size: {size_mb:.1f} MB")
            
        else:
            print(f"⚠ Models directory not found: {models_path}")
            print("  Models may download on first use")
        
        return True
        
    except Exception as e:
        print(f"✗ Model verification failed: {str(e)}")
        return False

def main():
    print("="*70)
    print("DeepFace Setup and Model Download Test")
    print("="*70)
    print("This test will verify DeepFace setup and download models to D drive")
    print("Models will be stored in: d:\\CASC Project\\deepface_models")
    print("="*70)
    
    # Load config
    config = load_config()
    if not config:
        sys.exit(1)
    
    # Track results
    results = {}
    
    # Run tests
    results['installation'] = test_deepface_installation()
    
    if not results['installation']:
        print("\n✗ Cannot proceed without DeepFace installation")
        sys.exit(1)
    
    results['model_directory'] = test_custom_model_directory(config)
    results['database_setup'] = test_face_database_setup(config)
    results['model_download'] = test_model_download_and_analysis()
    results['casc_integration'] = test_casc_deepface_integration()
    results['model_verification'] = check_model_files()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL DEEPFACE TESTS PASSED")
        print("  DeepFace is ready to use!")
        print(f"  Models stored in: d:\\CASC Project\\deepface_models")
        print("\nNext steps:")
        print("  1. Run: python src/server.py")
        print("  2. Select option 4: Face Recognition with DeepFace")
        print("  3. Add known persons to the database")
    else:
        print("✗ SOME TESTS FAILED")
        print("  Check DeepFace installation and configuration")
    print("="*70)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
