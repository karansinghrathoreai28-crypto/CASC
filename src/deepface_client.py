from deepface import DeepFace
import cv2
import os
import pandas as pd
from datetime import datetime
import json

class DeepFaceClient:
    def __init__(self, config):
        self.config = config
        self.db_path = config['deepface']['db_path']
        self.model_name = config['deepface']['model_name']
        self.detector_backend = config['deepface']['detector_backend']
        self.distance_metric = config['deepface']['distance_metric']
        
        # Set custom model directory
        self.models_path = "d:\\CASC Project\\deepface_models"
        os.makedirs(self.models_path, exist_ok=True)
        
        # Set environment variable to change DeepFace model directory
        os.environ['DEEPFACE_HOME'] = self.models_path
        
        # Create database directory if it doesn't exist
        os.makedirs(self.db_path, exist_ok=True)
        
        print(f"DeepFace client initialized")
        print(f"  Database: {self.db_path}")
        print(f"  Models Directory: {self.models_path}")
        print(f"  Model: {self.model_name}")
        print(f"  Detector: {self.detector_backend}")
        
        # Pre-load models to ensure they download to the correct location
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize and download models to custom directory"""
        try:
            print("Initializing DeepFace models...")
            
            # Create a dummy image for model initialization
            dummy_img = os.path.join(self.models_path, "dummy.jpg")
            if not os.path.exists(dummy_img):
                import numpy as np
                dummy_array = np.zeros((224, 224, 3), dtype=np.uint8)
                cv2.imwrite(dummy_img, dummy_array)
            
            # Initialize models by running a dummy analysis
            try:
                DeepFace.analyze(
                    img_path=dummy_img,
                    actions=['emotion'],
                    detector_backend=self.detector_backend,
                    enforce_detection=False,
                    silent=True
                )
                print("Models initialized successfully")
            except:
                print("Note: Some models may download on first use")
            
            # Clean up dummy image
            if os.path.exists(dummy_img):
                os.remove(dummy_img)
                
        except Exception as e:
            print(f"Model initialization warning: {str(e)}")
    
    def analyze_face(self, frame):
        """
        Analyze facial attributes: emotion, age, gender, race
        Returns: dict with analysis results
        """
        try:
            # Analyze the frame
            analysis = DeepFace.analyze(
                img_path=frame,
                actions=['emotion', 'age', 'gender', 'race'],
                detector_backend=self.detector_backend,
                enforce_detection=False,
                silent=True
            )
            
            # DeepFace returns a list of results (one per face)
            if isinstance(analysis, list):
                analysis = analysis[0] if analysis else {}
            
            # Extract relevant information
            result = {
                'face_detected': True,
                'dominant_emotion': analysis.get('dominant_emotion', 'unknown'),
                'emotion_scores': analysis.get('emotion', {}),
                'age': int(analysis.get('age', 0)),
                'gender': analysis.get('dominant_gender', 'unknown'),
                'gender_confidence': analysis.get('gender', {}),
                'dominant_race': analysis.get('dominant_race', 'unknown'),
                'race_scores': analysis.get('race', {}),
                'region': analysis.get('region', {})
            }
            
            return result
            
        except Exception as e:
            return {
                'face_detected': False,
                'error': str(e),
                'dominant_emotion': 'unknown',
                'age': 0,
                'gender': 'unknown',
                'dominant_race': 'unknown'
            }
    
    def identify_face(self, frame):
        """
        Identify face against known persons database
        Returns: dict with identification results
        """
        try:
            # Check if database has any known persons
            if not os.listdir(self.db_path):
                return {
                    'identified': False,
                    'reason': 'No known persons in database',
                    'confidence': 0
                }
            
            # Find matching faces
            dfs = DeepFace.find(
                img_path=frame,
                db_path=self.db_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                distance_metric=self.distance_metric,
                enforce_detection=False,
                silent=True
            )
            
            # Process results
            if isinstance(dfs, list) and len(dfs) > 0:
                df = dfs[0]
                
                if not df.empty:
                    # Get best match
                    best_match = df.iloc[0]
                    identity_path = best_match['identity']
                    distance = best_match['distance']
                    
                    # Extract person name from path
                    person_name = os.path.splitext(os.path.basename(identity_path))[0]
                    
                    # Calculate confidence (lower distance = higher confidence)
                    confidence = max(0, 1 - distance)
                    
                    # Threshold for positive identification
                    if confidence > 0.4:  # Adjust threshold as needed
                        return {
                            'identified': True,
                            'person_name': person_name,
                            'confidence': float(confidence),
                            'distance': float(distance),
                            'image_path': identity_path
                        }
            
            return {
                'identified': False,
                'reason': 'No match found',
                'confidence': 0
            }
            
        except Exception as e:
            return {
                'identified': False,
                'reason': f'Error: {str(e)}',
                'confidence': 0
            }
    
    def comprehensive_analysis(self, frame):
        """
        Complete analysis: identification + attributes
        """
        # Step 1: Try to identify the person
        identification = self.identify_face(frame)
        
        # Step 2: Analyze facial attributes
        attributes = self.analyze_face(frame)
        
        # Combine results
        result = {
            'timestamp': datetime.now().isoformat(),
            'identification': identification,
            'attributes': attributes,
            'is_known': identification.get('identified', False),
            'person_name': identification.get('person_name', 'Unknown'),
            'confidence': identification.get('confidence', 0)
        }
        
        # Generate human-readable summary
        if result['is_known']:
            summary = f"Known person: {result['person_name']} (confidence: {result['confidence']:.2f})"
        else:
            summary = f"Unknown person detected"
        
        if attributes['face_detected']:
            summary += f" - {attributes['gender']}, ~{attributes['age']} years old, {attributes['dominant_emotion']}"
        
        result['summary'] = summary
        
        return result
    
    def add_known_person(self, name, frame):
        """
        Add a new known person to the database
        Saves the image with the person's name
        """
        try:
            # Create filename
            filename = f"{name}.jpg"
            filepath = os.path.join(self.db_path, filename)
            
            # Check if person already exists
            if os.path.exists(filepath):
                return {
                    'success': False,
                    'error': f'Person "{name}" already exists in database'
                }
            
            # Verify face is detected
            analysis = self.analyze_face(frame)
            if not analysis['face_detected']:
                return {
                    'success': False,
                    'error': 'No face detected in image. Please try again with clearer image.'
                }
            
            # Save the image
            cv2.imwrite(filepath, frame)
            
            return {
                'success': True,
                'person_name': name,
                'filepath': filepath,
                'message': f'Successfully added {name} to known persons database'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def list_known_persons(self):
        """
        List all known persons in the database
        """
        try:
            persons = []
            
            for filename in os.listdir(self.db_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    name = os.path.splitext(filename)[0]
                    filepath = os.path.join(self.db_path, filename)
                    
                    persons.append({
                        'name': name,
                        'filepath': filepath,
                        'filename': filename
                    })
            
            return persons
            
        except Exception as e:
            print(f"Error listing persons: {str(e)}")
            return []
    
    def delete_person(self, name):
        """
        Delete a person from the database
        """
        try:
            filepath = os.path.join(self.db_path, f"{name}.jpg")
            
            if not os.path.exists(filepath):
                # Try other extensions
                for ext in ['.jpeg', '.png']:
                    alt_path = os.path.join(self.db_path, f"{name}{ext}")
                    if os.path.exists(alt_path):
                        filepath = alt_path
                        break
            
            if os.path.exists(filepath):
                os.remove(filepath)
                return {
                    'success': True,
                    'message': f'Successfully deleted {name}'
                }
            else:
                return {
                    'success': False,
                    'error': f'Person "{name}" not found in database'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def start_realtime_stream(self, db_path=None):
        """
        Start real-time face recognition stream
        This is for live demonstration mode
        """
        try:
            if db_path is None:
                db_path = self.db_path
            
            print("\nStarting real-time face recognition...")
            print("Press 'q' to quit")
            
            # Use DeepFace's built-in streaming
            DeepFace.stream(
                db_path=db_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                distance_metric=self.distance_metric,
                enable_face_analysis=True,
                time_threshold=2
            )
            
        except Exception as e:
            print(f"Error in real-time stream: {str(e)}")
    
    def assess_threat_level(self, analysis_result):
        """
        Assess threat level based on face analysis
        Returns: dict with threat assessment
        """
        is_known = analysis_result['is_known']
        attributes = analysis_result['attributes']
        
        # Base threat level
        if is_known:
            threat_level = 'LOW'
            threat_reason = f"Authorized person: {analysis_result['person_name']}"
            is_suspicious = False
        else:
            # Unknown person - analyze attributes for threat assessment
            emotion = attributes.get('dominant_emotion', 'unknown')
            age = attributes.get('age', 0)
            
            # Suspicious emotions
            suspicious_emotions = ['angry', 'fear', 'disgust']
            
            if emotion in suspicious_emotions:
                threat_level = 'HIGH'
                threat_reason = f"Unknown person displaying {emotion} emotion"
                is_suspicious = True
            else:
                threat_level = 'MEDIUM'
                threat_reason = "Unknown person detected with neutral/calm demeanor"
                is_suspicious = True
        
        return {
            'threat_level': threat_level,
            'is_suspicious': is_suspicious,
            'threat_reason': threat_reason,
            'requires_alert': is_suspicious
        }
