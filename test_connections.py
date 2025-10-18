"""
CASC System Connection Tester
Tests all API connections and components before running the main application
"""

import yaml
import sys
import os
from io import BytesIO

def load_config():
    """Load configuration file"""
    try:
        config_path = "config/config.yaml"
        if not os.path.exists(config_path):
            config_path = "d:\\CASC Project\\config\\config.yaml"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("✓ Configuration file loaded successfully")
        return config
    except Exception as e:
        print(f"✗ Failed to load configuration: {str(e)}")
        return None

def test_opencv():
    """Test OpenCV installation and camera access"""
    print("\n--- Testing OpenCV ---")
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
        
        # Test camera access
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✓ Camera access successful (Frame shape: {frame.shape})")
            else:
                print("✗ Could not read frame from camera")
            cap.release()
        else:
            print("⚠ Warning: Could not open camera (may not be available)")
        
        return True
    except Exception as e:
        print(f"✗ OpenCV test failed: {str(e)}")
        return False

def test_azure_vision(config):
    """Test Azure Computer Vision API connection"""
    print("\n--- Testing Azure Computer Vision ---")
    try:
        from azure.cognitiveservices.vision.computervision import ComputerVisionClient
        from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
        from msrest.authentication import CognitiveServicesCredentials
        import numpy as np
        import cv2
        
        endpoint = config['azure_vision']['endpoint']
        api_key = config['azure_vision']['api_key']
        
        if not endpoint or endpoint == "YOUR_AZURE_VISION_ENDPOINT":
            print("✗ Azure Vision endpoint not configured")
            return False
        
        if not api_key or api_key == "YOUR_AZURE_VISION_KEY":
            print("✗ Azure Vision API key not configured")
            return False
        
        print(f"  Endpoint: {endpoint}")
        
        # Create client
        credentials = CognitiveServicesCredentials(api_key)
        client = ComputerVisionClient(endpoint, credentials)
        print("✓ Azure Vision client created")
        
        # Create a test image (simple colored square)
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        test_image[:100, :] = [255, 0, 0]  # Blue top half
        test_image[100:, :] = [0, 255, 0]  # Green bottom half
        
        _, buffer = cv2.imencode('.jpg', test_image)
        image_stream = BytesIO(buffer.tobytes())
        
        # Test API call
        print("  Testing API call with sample image...")
        analysis = client.analyze_image_in_stream(
            image_stream,
            visual_features=[VisualFeatureTypes.description]
        )
        
        if analysis.description.captions:
            print(f"✓ Azure Vision API working! Sample result: '{analysis.description.captions[0].text}'")
            return True
        else:
            print("⚠ API responded but no description generated")
            return True
            
    except Exception as e:
        print(f"✗ Azure Vision test failed: {str(e)}")
        return False

def test_openrouter(config):
    """Test OpenRouter API connection"""
    print("\n--- Testing OpenRouter API ---")
    try:
        from openai import OpenAI
        
        api_key = config['openrouter']['api_key']
        model = config['openrouter']['model']
        
        if not api_key or api_key == "YOUR_OPENROUTER_API_KEY":
            print("✗ OpenRouter API key not configured")
            return False
        
        print(f"  Model: {model}")
        
        # Create OpenAI client with OpenRouter base URL
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        print("✓ OpenRouter client created")
        
        # Test API call
        print("  Testing API call...")
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://github.com/casc-project",
                "X-Title": "CASC Security Cam Test",
            },
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Say 'Connection successful' if you receive this message."
                }
            ]
        )
        
        answer = completion.choices[0].message.content
        print(f"✓ OpenRouter API working! Response: '{answer[:50]}...'")
        return True
            
    except Exception as e:
        print(f"✗ OpenRouter test failed: {str(e)}")
        return False

def test_cosmos_db(config):
    """Test Azure Cosmos DB connection"""
    print("\n--- Testing Azure Cosmos DB ---")
    try:
        from azure.cosmos import CosmosClient, PartitionKey, exceptions
        
        endpoint = config['cosmos_db']['endpoint']
        key = config['cosmos_db']['key']
        database_name = config['cosmos_db']['database_name']
        container_name = config['cosmos_db']['container_name']
        
        if not endpoint or endpoint == "YOUR_COSMOS_DB_ENDPOINT":
            print("✗ Cosmos DB endpoint not configured")
            return False
        
        if not key or key == "YOUR_COSMOS_DB_KEY":
            print("✗ Cosmos DB key not configured")
            return False
        
        print(f"  Endpoint: {endpoint}")
        print(f"  Database: {database_name}")
        print(f"  Container: {container_name}")
        
        # Create client
        client = CosmosClient(endpoint, key)
        print("✓ Cosmos DB client created")
        
        # Test database access
        database = client.create_database_if_not_exists(id=database_name)
        print(f"✓ Database '{database_name}' accessible")
        
        # Test container access (without throughput for serverless)
        try:
            container = database.create_container_if_not_exists(
                id=container_name,
                partition_key=PartitionKey(path="/event_id")
            )
            print(f"✓ Container '{container_name}' accessible (serverless mode)")
        except exceptions.CosmosHttpResponseError as e:
            if "throughput" in str(e).lower() or "serverless" in str(e).lower():
                print("ℹ Detected serverless account - container created without throughput")
                container = database.get_container_client(container_name)
            else:
                raise e
        
        # Test write operation
        test_doc = {
            'id': 'test-connection-doc',
            'event_id': 'test-connection-doc',
            'type': 'test',
            'message': 'Connection test successful'
        }
        
        container.upsert_item(body=test_doc)
        print("✓ Write operation successful")
        
        # Test read operation
        read_doc = container.read_item(
            item='test-connection-doc',
            partition_key='test-connection-doc'
        )
        print("✓ Read operation successful")
        
        # Clean up test document
        container.delete_item(
            item='test-connection-doc',
            partition_key='test-connection-doc'
        )
        print("✓ Test document cleaned up")
        
        return True
        
    except Exception as e:
        print(f"✗ Cosmos DB test failed: {str(e)}")
        return False

def test_all_dependencies():
    """Test all required Python packages"""
    print("\n--- Testing Python Dependencies ---")
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'azure.cognitiveservices.vision.computervision': 'azure-cognitiveservices-vision-computervision',
        'azure.cosmos': 'azure-cosmos',
        'requests': 'requests',
        'yaml': 'pyyaml',
        'PIL': 'pillow',
        'openai': 'openai'
    }
    
    all_installed = True
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT INSTALLED")
            all_installed = False
    
    return all_installed

def main():
    print("="*60)
    print("CASC System Connection Tester")
    print("="*60)
    
    # Track results
    results = {}
    
    # Test dependencies
    results['dependencies'] = test_all_dependencies()
    
    # Load config
    config = load_config()
    if not config:
        print("\n✗ Cannot proceed without configuration file")
        sys.exit(1)
    
    # Run all tests
    results['opencv'] = test_opencv()
    results['azure_vision'] = test_azure_vision(config)
    results['openrouter'] = test_openrouter(config)
    results['cosmos_db'] = test_cosmos_db(config)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED - System ready to use!")
        print("  Run 'python src/main.py' to start the application")
    else:
        print("✗ SOME TESTS FAILED - Please fix the issues above")
        print("  Check your config/config.yaml file and API credentials")
    print("="*60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
