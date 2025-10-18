# CASC - Contextual Aware Security Cam

An intelligent security camera system that provides natural language descriptions of security events and allows interactive Q&A.

## Features

- **Motion Detection**: Uses OpenCV to detect meaningful motion
- **Contextual Analysis**: Azure AI Vision analyzes what's happening
- **Face Recognition**: Azure Face API identifies known vs unknown persons
- **Emotion Detection**: Detects facial attributes and quality
- **Person Database**: Manage known persons for identification
- **Natural Language**: AI-generated summaries in plain English
- **Interactive Q&A**: Ask questions about security events
- **Event Storage**: Azure Cosmos DB stores all events and conversations

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Credentials
Edit `config/config.yaml` and add your API keys:
- Azure AI Vision endpoint and key
- OpenRouter API key
- Azure Cosmos DB endpoint and key

### 3. Test Connections

**Run all tests:**
```bash
python test_connections.py
```

**Test individual components:**

OpenRouter API only:
```bash
python test_openrouter.py
```

Azure Cosmos DB only:
```bash
python test_database.py
```

Azure Face API only:
```bash
python test_azure_face.py
```

These tests will verify:
- **test_connections.py**: All components (quick overview)
- **test_openrouter.py**: OpenRouter API, multiple models, Q&A functionality
- **test_database.py**: Cosmos DB CRUD operations, queries, performance
- **test_azure_face.py**: Face detection, recognition, person management

### 4. Run the Application
```bash
cd src
python main.py
```

## Usage

1. **Start Monitoring**: The system will continuously monitor the camera feed
2. **Motion Detection**: When motion is detected, the system:
   - Captures the frame
   - Analyzes it with Azure Vision
   - Generates a contextual summary
   - Saves everything to the database
3. **Ask Questions**: After each event, you can ask questions like:
   - "How many people were detected?"
   - "What objects were in the frame?"
   - "Was this a false alarm?"

## Configuration

- Adjust motion sensitivity in `config.yaml`
- Change cooldown period between detections
- Toggle image storage on/off
- Configure different AI models

## Troubleshooting

If `test_connections.py` shows failures:

1. **Dependencies Failed**: Run `pip install -r requirements.txt` again
2. **Camera Failed**: Check if another app is using the camera
3. **Azure Vision Failed**: Verify endpoint URL and API key
4. **OpenRouter Failed**: Check API key and model availability
5. **Cosmos DB Failed**: Verify endpoint, key, and firewall rules

### Camera Not Opening

If the camera doesn't open, run these tests in order:

1. **Basic camera test:**
```bash
python test_camera.py
```

2. **Live detector test:**
```bash
python src/live_detector.py
```

3. **Full system test:**
```bash
python src/server.py
```

**Common camera issues:**
- Another app is using the camera (close Zoom, Teams, etc.)
- Camera permissions not granted (check Windows Settings > Privacy > Camera)
- Wrong camera index (try changing `source: 0` to `source: 1` in config.yaml)

### Detection Not Working

**Test which detectors work for you:**
```bash
python test_detection_comparison.py
```

Press 'p' during the test to print detection counts.

**Common issues with Haar Cascade body detection:**

1. **Full body detector very unreliable** - This is a known limitation
   - Requires person to be 6-10 feet from camera
   - Person must be fully visible and upright
   - Often fails in typical webcam scenarios

2. **Upper body detector more reliable**
   - Works at closer range
   - Better for desk/room monitoring
   - Now enabled by default in the system

3. **Face detection most reliable**
   - Works at any reasonable distance
   - Good for identifying presence
   - Falls back to this if body detection fails

**Tips for better detection:**
- Ensure good lighting
- Stand upright facing camera
- Remove clutter from background
- For body detection, stand further back (6-10 feet)
- Upper body detection works best for typical webcam setups

## Architecture

