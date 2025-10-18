import yaml
import os
import cv2
from datetime import datetime
from live_detector import LiveDetector
from azure_vision_client import AzureVisionClient
from openrouter_client import OpenRouterClient
from database_manager import DatabaseManager
from azure_face_client import AzureFaceClient
from deepface_client import DeepFaceClient
import time

class CASCApplication:
    def __init__(self, config_path):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        print("Initializing CASC system...")
        self.live_detector = LiveDetector(self.config)
        self.vision_client = AzureVisionClient(self.config)
        self.openrouter_client = OpenRouterClient(self.config)
        self.database_manager = DatabaseManager(self.config)
        self.face_client = AzureFaceClient(self.config)
        self.deepface_client = DeepFaceClient(self.config)
        
        # Create image directory if needed
        if self.config['storage']['save_images']:
            os.makedirs(self.config['storage']['image_directory'], exist_ok=True)
        
        self.current_event_id = None
        print("CASC system initialized successfully!")
    
    def analyze_and_classify_threat(self, vision_analysis, faces_count, bodies_count):
        """Use AI to analyze if detection is suspicious"""
        prompt = f"""Analyze this security camera detection and determine if it's suspicious:

Detection Details:
- Faces detected: {faces_count}
- People/Bodies detected: {bodies_count}
- Scene description: {vision_analysis['description']}
- Objects: {', '.join([obj['name'] for obj in vision_analysis['objects']])}
- Tags: {', '.join(vision_analysis['tags'][:5])}

Based on this information:
1. Is this suspicious? (Yes/No)
2. Alert level: LOW, MEDIUM, or HIGH
3. Brief reason (one sentence)

Format your response as:
SUSPICIOUS: [Yes/No]
ALERT_LEVEL: [LOW/MEDIUM/HIGH]
REASON: [Your explanation]"""

        response = self.openrouter_client._call_api(
            prompt, 
            system_message="You are a security AI assistant. Analyze detections and classify threat levels."
        )
        
        return self._parse_threat_analysis(response)
    
    def _parse_threat_analysis(self, response):
        """Parse AI threat analysis response"""
        lines = response.strip().split('\n')
        result = {
            'is_suspicious': False,
            'alert_level': 'LOW',
            'reason': 'Normal activity detected'
        }
        
        for line in lines:
            if 'SUSPICIOUS:' in line:
                result['is_suspicious'] = 'yes' in line.lower()
            elif 'ALERT_LEVEL:' in line:
                if 'HIGH' in line.upper():
                    result['alert_level'] = 'HIGH'
                elif 'MEDIUM' in line.upper():
                    result['alert_level'] = 'MEDIUM'
                else:
                    result['alert_level'] = 'LOW'
            elif 'REASON:' in line:
                result['reason'] = line.split('REASON:', 1)[1].strip()
        
        return result
    
    def generate_alert_message(self, threat_analysis, vision_analysis, faces_count, bodies_count):
        """Generate alert message"""
        # Remove emojis from alert levels
        alert = f"""
{'='*70}
SECURITY ALERT - {threat_analysis['alert_level']} PRIORITY
{'='*70}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Status: {'SUSPICIOUS' if threat_analysis['is_suspicious'] else 'Normal'}

Detection Summary:
- Faces Detected: {faces_count}
- People Detected: {bodies_count}
- Scene: {vision_analysis['description']}

AI Analysis:
{threat_analysis['reason']}

{'='*70}
"""
        return alert
    
    def run_live_monitoring(self):
        """Main live monitoring with 15-second intervals"""
        print("\n" + "="*70)
        print("STARTING LIVE MONITORING MODE")
        print("="*70)
        print(f"Detection interval: {self.config['camera'].get('detection_interval', 15)} seconds")
        print("Press 'q' in the video window to quit")
        print("Video window will open shortly...")
        print("="*70 + "\n")
        
        self.live_detector.start_camera()
        
        # Give camera time to initialize
        import time
        time.sleep(1)
        
        try:
            for frame, faces, bodies in self.live_detector.process_live_stream():
                faces_count = len(faces)
                bodies_count = len(bodies)
                
                print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
                print(f"Detection: {faces_count} faces, {bodies_count} people")
                
                # Analyze with Azure Vision
                print("Analyzing with Azure Vision...")
                vision_analysis = self.vision_client.analyze_image(frame)
                
                # Analyze faces with Azure Face
                print("Analyzing faces with Azure Face...")
                face_analysis = self.face_client.analyze_frame_comprehensive(frame)
                print(f"Face Analysis: {face_analysis['summary']}")
                
                if vision_analysis:
                    # Classify threat level
                    print("AI threat analysis in progress...")
                    threat_analysis = self.analyze_and_classify_threat_with_faces(
                        vision_analysis, faces_count, bodies_count, face_analysis
                    )
                    
                    # Generate and display alert
                    alert_message = self.generate_alert_message(
                        threat_analysis, vision_analysis, faces_count, bodies_count
                    )
                    print(alert_message)
                    
                    # Generate detailed summary
                    ai_summary = self.openrouter_client.generate_context_summary(vision_analysis)
                    print(f"Detailed Summary:\n{ai_summary}\n")
                    
                    # Save to database with face information
                    print("Saving event to database...")
                    event_data = {
                        'faces_count': faces_count,
                        'bodies_count': bodies_count,
                        'threat_analysis': threat_analysis,
                        'alert_message': alert_message,
                        'face_analysis': face_analysis
                    }
                    
                    self.current_event_id = self.database_manager.save_event(
                        frame, vision_analysis, ai_summary, 
                        motion_level=faces_count + bodies_count
                    )
                    
                    if self.current_event_id:
                        print(f"Event saved with ID: {self.current_event_id}")
                        
                        # Save image locally
                        if self.config['storage']['save_images']:
                            image_path = os.path.join(
                                self.config['storage']['image_directory'],
                                f"{self.current_event_id}.jpg"
                            )
                            cv2.imwrite(image_path, frame)
                            print(f"Image saved: {image_path}")
                        
                        # Interactive Q&A for suspicious alerts - NON-BLOCKING
                        if threat_analysis['is_suspicious'] or threat_analysis['alert_level'] in ['MEDIUM', 'HIGH']:
                            print("\n" + "="*70)
                            print("SUSPICIOUS ACTIVITY DETECTED")
                            print("="*70)
                            print("You can ask questions, or press ENTER to continue monitoring")
                            print("Type 'skip' and press ENTER to continue immediately")
                            print("="*70)
                            
                            # Start Q&A without blocking video
                            self.interactive_qa_with_video(vision_analysis, ai_summary, event_data)
                
                print("-"*70)
                
        except KeyboardInterrupt:
            print("\n\nStopping live monitoring...")
        finally:
            self.live_detector.release()
    
    def interactive_qa_with_video(self, vision_analysis, ai_summary, event_data=None):
        """Q&A that keeps video window alive"""
        event_context = {
            'ai_summary': ai_summary,
            'vision_analysis': vision_analysis,
            'event_id': self.current_event_id
        }
        
        if event_data:
            event_context.update(event_data)
        
        print("\nQ&A Mode (video continues in background)")
        print("Type your question or 'skip' to continue monitoring\n")
        
        # Simple non-blocking input
        question = input("Your question (or skip): ").strip()
        
        if question.lower() == 'skip' or not question:
            print("Continuing monitoring...\n")
            return
        
        # Process the question
        print("Thinking...")
        answer = self.openrouter_client.answer_question(question, event_context)
        
        # Check if answer is empty
        if not answer or answer.strip() == "":
            print("\nAnswer: [ERROR: Empty response from AI]")
            print("This may indicate:")
            print("1. API key issue")
            print("2. Model not responding")
            print("3. Network connectivity problem")
            print("\nTry running: python test_openrouter.py to diagnose")
        else:
            print(f"\nAnswer: {answer}\n")
        
        # Save conversation to database only if answer exists
        if self.current_event_id and answer and answer.strip():
            self.database_manager.add_conversation(
                self.current_event_id, question, answer
            )
        
        # Ask if they want to continue Q&A
        continue_qa = input("Ask another question? (y/n): ").strip().lower()
        if continue_qa == 'y':
            self.interactive_qa_with_video(vision_analysis, ai_summary, event_data)

    # Remove or update the old interactive_qa_non_blocking method
    def interactive_qa(self, vision_analysis, ai_summary, event_data=None):
        """Legacy method - redirects to new implementation"""
        self.interactive_qa_with_video(vision_analysis, ai_summary, event_data)

    def view_recent_events(self, limit=5):
        """View recent security events"""
        print("\n" + "="*70)
        print(f"RECENT {limit} SECURITY EVENTS")
        print("="*70)
        
        events = self.database_manager.get_recent_events(limit)
        
        if not events:
            print("No events found.")
            return
        
        for i, event in enumerate(events, 1):
            print(f"\n{i}. Event ID: {event['id']}")
            print(f"   Time: {event['timestamp']}")
            print(f"   Summary: {event['ai_summary'][:100]}...")
            print(f"   Conversations: {len(event.get('conversations', []))}")
            print("-"*70)

    def interactive_event_qa(self):
        """Ask questions about a specific saved event"""
        print("\n" + "="*70)
        print("INTERACTIVE EVENT Q&A")
        print("="*70)
        
        # Get recent events
        events = self.database_manager.get_recent_events(10)
        
        if not events:
            print("No events found in database.")
            input("\nPress ENTER to return to main menu...")
            return
        
        # Display events
        print("\nRecent Events:")
        print("-"*70)
        for i, event in enumerate(events, 1):
            timestamp = event['timestamp']
            summary = event['ai_summary'][:80] + "..." if len(event['ai_summary']) > 80 else event['ai_summary']
            print(f"{i}. [{timestamp}] {summary}")
        print("-"*70)
        
        # Select event
        try:
            choice = input(f"\nSelect event (1-{len(events)}) or 'back' to return: ").strip()
            
            if choice.lower() == 'back':
                return
            
            event_index = int(choice) - 1
            if event_index < 0 or event_index >= len(events):
                print("Invalid selection.")
                input("\nPress ENTER to return...")
                return
            
            selected_event = events[event_index]
            
        except ValueError:
            print("Invalid input.")
            input("\nPress ENTER to return...")
            return
        
        # Display event details
        print("\n" + "="*70)
        print("EVENT DETAILS")
        print("="*70)
        print(f"Event ID: {selected_event['id']}")
        print(f"Timestamp: {selected_event['timestamp']}")
        print(f"Motion Level: {selected_event.get('motion_level', 'N/A')}")
        print("\nAI Summary:")
        print(selected_event['ai_summary'])
        
        if selected_event.get('vision_analysis'):
            vision = selected_event['vision_analysis']
            print(f"\nScene Description: {vision.get('description', 'N/A')}")
            print(f"Detected Objects: {', '.join([obj['name'] for obj in vision.get('objects', [])])}")
            print(f"Tags: {', '.join(vision.get('tags', [])[:5])}")
        
        # Show existing conversations
        conversations = selected_event.get('conversations', [])
        if conversations:
            print(f"\nPrevious Conversations ({len(conversations)}):")
            print("-"*70)
            for i, conv in enumerate(conversations, 1):
                print(f"\nQ{i}: {conv['question']}")
                print(f"A{i}: {conv['answer']}")
        
        print("="*70)
        
        # Q&A loop
        event_context = {
            'event_id': selected_event['id'],
            'ai_summary': selected_event['ai_summary'],
            'vision_analysis': selected_event.get('vision_analysis', {}),
            'timestamp': selected_event['timestamp'],
            'motion_level': selected_event.get('motion_level', 0)
        }
        
        print("\n" + "="*70)
        print("ASK QUESTIONS ABOUT THIS EVENT")
        print("="*70)
        print("Type your questions about this security event")
        print("Commands: 'done' (return to menu) | 'show' (show event details again)")
        print("="*70 + "\n")
        
        while True:
            question = input("Your question: ").strip()
            
            if question.lower() == 'done':
                print("\nReturning to main menu...\n")
                break
            elif question.lower() == 'show':
                print(f"\n{selected_event['ai_summary']}\n")
                continue
            elif question.lower() == 'debug':
                # Show full event data for debugging
                print("\n--- DEBUG: Event Data ---")
                print(f"Event ID: {selected_event['id']}")
                print(f"Timestamp: {selected_event['timestamp']}")
                print(f"Vision Analysis: {selected_event.get('vision_analysis', {})}")
                print(f"AI Summary: {selected_event.get('ai_summary', 'N/A')}")
                print("--- END DEBUG ---\n")
                continue
            elif not question:
                continue
            
            # Show what's being sent to the AI
            print(f"Thinking... ")
            
            # Process question
            answer = self.openrouter_client.answer_question(question, event_context)
            
            # Check if answer is empty
            if not answer or answer.strip() == "":
                print("\nAnswer: [ERROR: Empty response from AI]")
                print("\nTroubleshooting:")
                print("1. Your OpenRouter API key may be invalid or expired")
                print("2. Free tier rate limits may be exceeded")
                print("3. Try a different model in config.yaml")
                print("4. Check OpenRouter dashboard: https://openrouter.ai/")
                print("5. Type 'debug' to see raw event data")
            else:
                print(f"\nAnswer: {answer}\n")
            
            # Save conversation to database only if answer exists
            if answer and answer.strip():
                self.database_manager.add_conversation(
                    selected_event['id'], question, answer
                )
                print("(Conversation saved to database)")
            else:
                print("(Conversation NOT saved due to empty response)")

    def analyze_and_classify_threat_with_faces(self, vision_analysis, faces_count, bodies_count, face_analysis):
        """Enhanced threat analysis including face recognition"""
        unknown_persons = face_analysis.get('unknown_count', 0)
        known_persons = face_analysis.get('known_persons', [])
        
        prompt = f"""Analyze this security camera detection and determine if it's suspicious:

Detection Details:
- Faces detected: {faces_count}
- People/Bodies detected: {bodies_count}
- Known persons: {', '.join(known_persons) if known_persons else 'None'}
- Unknown persons: {unknown_persons}
- Scene description: {vision_analysis['description']}
- Objects: {', '.join([obj['name'] for obj in vision_analysis['objects']])}
- Tags: {', '.join(vision_analysis['tags'][:5])}

Based on this information:
1. Is this suspicious? (Yes/No)
2. Alert level: LOW, MEDIUM, or HIGH
3. Brief reason (one sentence)

Format your response as:
SUSPICIOUS: [Yes/No]
ALERT_LEVEL: [LOW/MEDIUM/HIGH]
REASON: [Your explanation]"""

        response = self.openrouter_client._call_api(
            prompt,
            system_message="You are a security AI assistant. Unknown persons should raise alert level. Analyze detections and classify threat levels."
        )
        
        return self._parse_threat_analysis(response)
    
    def manage_known_persons(self):
        """Menu for managing known persons database"""
        while True:
            print("\n" + "="*70)
            print("KNOWN PERSONS MANAGEMENT")
            print("="*70)
            print("1. List Known Persons")
            print("2. Add New Person (from camera)")
            print("3. Delete Person")
            print("4. Back to Main Menu")
            print("="*70)
            
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == '1':
                self._list_known_persons()
            elif choice == '2':
                self._add_known_person_from_camera()
            elif choice == '3':
                self._delete_known_person()
            elif choice == '4':
                break
    
    def _list_known_persons(self):
        """List all known persons"""
        persons = self.face_client.list_known_persons()
        
        if not persons:
            print("\nNo known persons in database.")
            return
        
        print("\nKnown Persons:")
        print("-"*70)
        for i, person in enumerate(persons, 1):
            print(f"{i}. {person['name']} (ID: {person['person_id'][:8]}..., Faces: {person['face_count']})")
        print("-"*70)
        input("\nPress ENTER to continue...")
    
    def _add_known_person_from_camera(self):
        """Capture photo from camera and add person"""
        print("\n" + "="*70)
        print("ADD KNOWN PERSON")
        print("="*70)
        
        name = input("Enter person's name: ").strip()
        if not name:
            print("Name cannot be empty.")
            return
        
        print("\nPreparing camera...")
        self.live_detector.start_camera()
        
        print("Position the person in front of the camera.")
        print("Press SPACE to capture, ESC to cancel")
        
        # Create window explicitly
        cv2.namedWindow("Capture Photo", cv2.WINDOW_NORMAL)
        
        captured_frame = None
        while True:
            ret, frame = self.live_detector.cap.read()
            if not ret:
                print("Error: Cannot read from camera")
                break
            
            # Add instructions to the frame
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Capturing: {name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press SPACE to capture", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, "Press ESC to cancel", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw a box showing where face should be
            height, width = display_frame.shape[:2]
            center_x, center_y = width // 2, height // 2
            box_size = 200
            cv2.rectangle(display_frame, 
                         (center_x - box_size, center_y - box_size),
                         (center_x + box_size, center_y + box_size),
                         (0, 255, 0), 2)
            cv2.putText(display_frame, "Position face here", 
                       (center_x - box_size, center_y - box_size - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show preview
            cv2.imshow("Capture Photo", display_frame)
            
            # Wait for key press with longer delay for better responsiveness
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32:  # SPACE key
                captured_frame = frame.copy()
                print("\nPhoto captured!")
                break
            elif key == 27:  # ESC key
                print("\nCapture cancelled.")
                break
        
        # Close the window
        cv2.destroyWindow("Capture Photo")
        cv2.waitKey(1)  # Process window events
        
        if captured_frame is not None:
            print("Adding person to database...")
            result = self.face_client.add_known_person(name, frame=captured_frame)
            
            if result['success']:
                print(f"SUCCESS: {name} added to known persons database!")
            else:
                print(f"ERROR: {result.get('error', 'Unknown error')}")
        else:
            print("Capture cancelled.")
        
        input("\nPress ENTER to continue...")
    
    def _delete_known_person(self):
        """Delete a person from database"""
        persons = self.face_client.list_known_persons()
        
        if not persons:
            print("\nNo known persons to delete.")
            input("\nPress ENTER to continue...")
            return
        
        print("\nKnown Persons:")
        for i, person in enumerate(persons, 1):
            print(f"{i}. {person['name']}")
        
        try:
            choice = int(input(f"\nSelect person to delete (1-{len(persons)}): "))
            if 1 <= choice <= len(persons):
                selected = persons[choice - 1]
                confirm = input(f"Delete {selected['name']}? (yes/no): ").strip().lower()
                
                if confirm == 'yes':
                    result = self.face_client.delete_person(selected['person_id'])
                    if result['success']:
                        print(f"SUCCESS: {selected['name']} deleted.")
                    else:
                        print(f"ERROR: {result.get('error')}")
        except ValueError:
            print("Invalid input.")
        
        input("\nPress ENTER to continue...")
    
    def run_live_monitoring_with_deepface(self):
        """Live monitoring with motion-first strategy"""
        print("\n" + "="*70)
        print("STARTING LIVE MONITORING WITH MOTION-FIRST STRATEGY")
        print("="*70)
        print("Strategy:")
        print("  1. Camera always on (local processing)")
        print("  2. Motion detected? → Check for faces")
        print("  3. Face found? → Send to Azure + DeepFace + OpenRouter")
        print("  4. No motion/face? → Skip Azure (save API costs)")
        print(f"  5. Recheck every {self.config['camera'].get('detection_interval', 15)} seconds")
        print("="*70 + "\n")
        
        self.live_detector.start_camera()
        
        import time
        time.sleep(1)
        
        # Track API calls
        api_call_count = 0
        skipped_frames = 0
        
        try:
            for frame, faces, hands_count in self.live_detector.process_live_stream():
                api_call_count += 1
                faces_count = len(faces)
                
                print(f"\n{'='*70}")
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] API CALL #{api_call_count}")
                print(f"{'='*70}")
                print(f"Motion detected → Faces found: {faces_count} → Sending to Azure...")
                
                # DeepFace comprehensive analysis
                print("1/3 Analyzing with DeepFace...")
                deepface_analysis = self.deepface_client.comprehensive_analysis(frame)
                print(f"    Result: {deepface_analysis['summary']}")
                
                # Assess threat level
                threat_assessment = self.deepface_client.assess_threat_level(deepface_analysis)
                print(f"    Threat Level: {threat_assessment['threat_level']}")
                
                # Analyze with Azure Vision
                print("2/3 Analyzing with Azure Vision...")
                vision_analysis = self.vision_client.analyze_image(frame)
                
                if vision_analysis:
                    print(f"    Scene: {vision_analysis['description']}")
                    
                    # Generate AI summary with OpenRouter
                    print("3/3 Generating summary with OpenRouter...")
                    ai_summary = self.generate_ai_summary_with_deepface(
                        vision_analysis, deepface_analysis, threat_assessment
                    )
                    
                    # Generate enhanced alert
                    alert_message = self.generate_deepface_alert(
                        threat_assessment, vision_analysis, deepface_analysis
                    )
                    print(alert_message)
                    print(f"AI Summary: {ai_summary}\n")
                    
                    # Save to database
                    print("Saving event to database...")
                    event_data = {
                        'faces_count': faces_count,
                        'hands_count': hands_count,
                        'deepface_analysis': deepface_analysis,
                        'threat_assessment': threat_assessment,
                        'alert_message': alert_message
                    }
                    
                    self.current_event_id = self.database_manager.save_event(
                        frame, vision_analysis, ai_summary,
                        motion_level=faces_count + hands_count
                    )
                    
                    if self.current_event_id:
                        print(f"Event saved with ID: {self.current_event_id}")
                        
                        # Save image
                        if self.config['storage']['save_images']:
                            image_path = os.path.join(
                                self.config['storage']['image_directory'],
                                f"{self.current_event_id}.jpg"
                            )
                            cv2.imwrite(image_path, frame)
                            print(f"Image saved: {image_path}")
                        
                        # Q&A for suspicious activity
                        if threat_assessment['is_suspicious']:
                            print("\n" + "="*70)
                            print(f"ALERT: {threat_assessment['threat_level']} PRIORITY")
                            print("="*70)
                            user_input = input("Ask questions about this alert? (y/n): ").strip().lower()
                            if user_input == 'y':
                                self.interactive_qa_with_video(vision_analysis, ai_summary, event_data)
                
                print(f"\nTotal API calls so far: {api_call_count}")
                print("="*70)
                
        except KeyboardInterrupt:
            print(f"\n\nStopping live monitoring...")
            print(f"Total API calls made: {api_call_count}")
        finally:
            self.live_detector.release()
    
    def generate_deepface_alert(self, threat_assessment, vision_analysis, deepface_analysis):
        """Generate alert with DeepFace information"""
        alert = f"""
{'='*70}
SECURITY ALERT - {threat_assessment['threat_level']} PRIORITY
{'='*70}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Status: {'SUSPICIOUS' if threat_assessment['is_suspicious'] else 'Normal'}

Detection Summary:
- Person: {deepface_analysis['person_name']}
- Confidence: {deepface_analysis['confidence']:.2%}
- Gender: {deepface_analysis['attributes']['gender']}
- Age: ~{deepface_analysis['attributes']['age']} years
- Emotion: {deepface_analysis['attributes']['dominant_emotion']}
- Scene: {vision_analysis['description']}

AI Analysis:
{threat_assessment['threat_reason']}

{'='*70}
"""
        return alert
    
    def generate_ai_summary_with_deepface(self, vision_analysis, deepface_analysis, threat_assessment):
        """Generate AI summary enriched with DeepFace data"""
        person_desc = f"{'Known person ' + deepface_analysis['person_name'] if deepface_analysis['is_known'] else 'Unknown person'}"
        
        prompt = f"""Based on this security camera analysis, provide a clear, detailed summary:

Person Information:
- Identity: {person_desc}
- Gender: {deepface_analysis['attributes']['gender']}
- Approximate Age: {deepface_analysis['attributes']['age']}
- Emotion: {deepface_analysis['attributes']['dominant_emotion']}
- Threat Level: {threat_assessment['threat_level']}

Scene Description: {vision_analysis['description']}
Objects detected: {', '.join([obj['name'] for obj in vision_analysis['objects']])}
Tags: {', '.join(vision_analysis['tags'][:5])}

Provide a natural language summary suitable for a security alert, incorporating the facial analysis details."""

        response = self.openrouter_client._call_api(
            prompt,
            system_message="You are a security AI assistant. Provide clear, factual summaries incorporating facial recognition data."
        )
        return response
    
    def manage_known_persons_deepface(self):
        """Manage known persons with DeepFace"""
        while True:
            print("\n" + "="*70)
            print("KNOWN PERSONS MANAGEMENT (DeepFace)")
            print("="*70)
            print("1. List Known Persons")
            print("2. Add New Person (from camera)")
            print("3. Delete Person")
            print("4. Test Recognition (live demo)")
            print("5. Back to Main Menu")
            print("="*70)
            
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == '1':
                self._list_known_persons_deepface()
            elif choice == '2':
                self._add_known_person_deepface()
            elif choice == '3':
                self._delete_known_person_deepface()
            elif choice == '4':
                self._test_recognition_deepface()
            elif choice == '5':
                break
    
    def _list_known_persons_deepface(self):
        """List all known persons"""
        persons = self.deepface_client.list_known_persons()
        
        if not persons:
            print("\nNo known persons in database.")
            print(f"Database location: {self.deepface_client.db_path}")
        else:
            print(f"\nKnown Persons ({len(persons)}):")
            print("-"*70)
            for i, person in enumerate(persons, 1):
                print(f"{i}. {person['name']}")
            print("-"*70)
        
        input("\nPress ENTER to continue...")
    
    def _add_known_person_deepface(self):
        """Add person using DeepFace"""
        print("\n" + "="*70)
        print("ADD KNOWN PERSON (DeepFace)")
        print("="*70)
        
        name = input("Enter person's name: ").strip()
        if not name:
            print("Name cannot be empty.")
            input("\nPress ENTER to continue...")
            return
        
        print("\nPreparing camera...")
        self.live_detector.start_camera()
        
        cv2.namedWindow("Capture Photo", cv2.WINDOW_NORMAL)
        
        print("Position face in front of camera.")
        print("Press SPACE to capture, ESC to cancel")
        
        captured_frame = None
        while True:
            ret, frame = self.live_detector.cap.read()
            if not ret:
                break
            
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Capturing: {name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press SPACE to capture", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, "Press ESC to cancel", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Capture Photo", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32:  # SPACE
                captured_frame = frame.copy()
                print("\nPhoto captured!")
                break
            elif key == 27:  # ESC
                print("\nCapture cancelled.")
                break
        
        cv2.destroyWindow("Capture Photo")
        cv2.waitKey(1)
        
        if captured_frame is not None:
            print("Adding person to DeepFace database...")
            result = self.deepface_client.add_known_person(name, captured_frame)
            
            if result['success']:
                print(f"SUCCESS: {result['message']}")
            else:
                print(f"ERROR: {result['error']}")
        
        input("\nPress ENTER to continue...")
    
    def _delete_known_person_deepface(self):
        """Delete person from DeepFace database"""
        persons = self.deepface_client.list_known_persons()
        
        if not persons:
            print("\nNo known persons to delete.")
            input("\nPress ENTER to continue...")
            return
        
        print("\nKnown Persons:")
        for i, person in enumerate(persons, 1):
            print(f"{i}. {person['name']}")
        
        try:
            choice = int(input(f"\nSelect person to delete (1-{len(persons)}): "))
            if 1 <= choice <= len(persons):
                selected = persons[choice - 1]
                confirm = input(f"Delete {selected['name']}? (yes/no): ").strip().lower()
                
                if confirm == 'yes':
                    result = self.deepface_client.delete_person(selected['name'])
                    if result['success']:
                        print(f"SUCCESS: {result['message']}")
                    else:
                        print(f"ERROR: {result['error']}")
        except ValueError:
            print("Invalid input.")
        
        input("\nPress ENTER to continue...")
    
    def _test_recognition_deepface(self):
        """Live demonstration mode with DeepFace"""
        print("\n" + "="*70)
        print("LIVE RECOGNITION TEST (DeepFace Stream)")
        print("="*70)
        print("This will open a live video feed with real-time recognition")
        print("Known persons will be labeled with their names")
        print("Unknown persons will be labeled as 'Unknown'")
        print("\nPress 'q' in the video window to exit")
        input("\nPress ENTER to start...")
        
        try:
            self.deepface_client.start_realtime_stream()
        except Exception as e:
            print(f"Error: {str(e)}")
        
        input("\nPress ENTER to continue...")

def main():
    config_path = "config/config.yaml"
    if not os.path.exists(config_path):
        config_path = "d:\\CASC Project\\config\\config.yaml"
    try:
        app = CASCApplication(config_path)
    except Exception as e:
        print(f"Failed to initialize CASC: {str(e)}")
        return
    while True:
        print("\n" + "="*70)
        print("CASC - Contextual Aware Security Cam")
        print("="*70)
        print("1. Start Live Monitoring (15-second detection)")
        print("2. View Recent Events")
        print("3. Ask Questions About Saved Events")
        print("4. Face Recognition with DeepFace")
        print("5. Exit")
        print("="*70)
        choice = input("\nSelect option (1-5): ").strip()
        if choice == '1':
            app.run_live_monitoring_with_deepface()
        elif choice == '2':
            app.view_recent_events()
        elif choice == '3':
            app.interactive_event_qa()
        elif choice == '4':
            app.manage_known_persons_deepface()
        elif choice == '5':
            print("\nGoodbye!")
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()
