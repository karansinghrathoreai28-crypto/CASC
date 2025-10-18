from azure.core.credentials import AzureKeyCredential
from azure.ai.vision.face import FaceAdministrationClient, FaceClient
from azure.ai.vision.face.models import (
    FaceAttributeTypeRecognition04,
    FaceDetectionModel,
    FaceRecognitionModel,
    QualityForRecognition
)
import cv2
from io import BytesIO
import time

class AzureFaceClient:
    def __init__(self, config):
        self.endpoint = config['azure_face']['endpoint']
        self.api_key = config['azure_face']['api_key']
        self.person_group_id = config['azure_face']['person_group_id']
        
        # Initialize clients
        self.face_admin_client = FaceAdministrationClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.api_key)
        )
        self.face_client = FaceClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.api_key)
        )
        
        # Initialize person group
        self._initialize_person_group()
        
        print("Azure Face client initialized successfully")
    
    def _initialize_person_group(self):
        """Create person group if it doesn't exist"""
        try:
            # Try to get existing person group
            self.face_admin_client.large_person_group.get(
                large_person_group_id=self.person_group_id
            )
            print(f"Person group '{self.person_group_id}' already exists")
        except Exception:
            # Create new person group
            try:
                self.face_admin_client.large_person_group.create(
                    large_person_group_id=self.person_group_id,
                    name="CASC Known Persons",
                    recognition_model=FaceRecognitionModel.RECOGNITION04
                )
                print(f"Created person group: {self.person_group_id}")
            except Exception as e:
                print(f"Warning: Could not create person group: {str(e)}")
    
    def detect_faces_with_attributes(self, frame):
        """
        Detect faces and extract attributes including emotions
        Returns: list of face details with emotions and quality
        """
        try:
            # Convert frame to bytes
            _, buffer = cv2.imencode('.jpg', frame)
            image_stream = BytesIO(buffer.tobytes())
            
            # Detect faces with attributes
            detected_faces = self.face_client.detect(
                image_content=image_stream.getvalue(),
                detection_model=FaceDetectionModel.DETECTION03,
                recognition_model=FaceRecognitionModel.RECOGNITION04,
                return_face_id=True,
                return_face_attributes=[
                    FaceAttributeTypeRecognition04.HEAD_POSE,
                    FaceAttributeTypeRecognition04.MASK,
                    FaceAttributeTypeRecognition04.BLUR,
                    FaceAttributeTypeRecognition04.EXPOSURE,
                    FaceAttributeTypeRecognition04.NOISE,
                    FaceAttributeTypeRecognition04.QUALITY_FOR_RECOGNITION
                ],
                return_face_landmarks=False,
                return_recognition_model=True,
                face_id_time_to_live=86400
            )
            
            faces_info = []
            for face in detected_faces:
                face_info = {
                    'face_id': face.face_id,
                    'rectangle': {
                        'left': face.face_rectangle.left,
                        'top': face.face_rectangle.top,
                        'width': face.face_rectangle.width,
                        'height': face.face_rectangle.height
                    },
                    'quality': face.face_attributes.quality_for_recognition.value if face.face_attributes.quality_for_recognition else 'UNKNOWN',
                    'mask': face.face_attributes.mask.type.value if face.face_attributes.mask else 'UNKNOWN',
                    'blur': face.face_attributes.blur.blur_level.value if face.face_attributes.blur else 'UNKNOWN'
                }
                faces_info.append(face_info)
            
            return faces_info
            
        except Exception as e:
            print(f"Error detecting faces: {str(e)}")
            return []
    
    def identify_faces(self, face_ids):
        """
        Identify faces against known persons database
        Returns: dict mapping face_id to person info
        """
        if not face_ids:
            return {}
        
        try:
            # Check if person group is trained
            training_status = self.face_admin_client.large_person_group.get_training_status(
                large_person_group_id=self.person_group_id
            )
            
            if training_status.status.value != 'succeeded':
                return {fid: {'identified': False, 'reason': 'Person group not trained'} for fid in face_ids}
            
            # Identify faces
            identify_results = self.face_client.identify_from_large_person_group(
                face_ids=face_ids,
                large_person_group_id=self.person_group_id,
                max_num_of_candidates_returned=1,
                confidence_threshold=0.5
            )
            
            identified_faces = {}
            for result in identify_results:
                if result.candidates:
                    candidate = result.candidates[0]
                    # Get person details
                    person = self.face_admin_client.large_person_group.get_person(
                        large_person_group_id=self.person_group_id,
                        person_id=candidate.person_id
                    )
                    
                    identified_faces[result.face_id] = {
                        'identified': True,
                        'person_name': person.name,
                        'person_id': candidate.person_id,
                        'confidence': candidate.confidence
                    }
                else:
                    identified_faces[result.face_id] = {
                        'identified': False,
                        'reason': 'Unknown person'
                    }
            
            return identified_faces
            
        except Exception as e:
            print(f"Error identifying faces: {str(e)}")
            return {fid: {'identified': False, 'reason': str(e)} for fid in face_ids}
    
    def add_known_person(self, name, image_path=None, frame=None):
        """
        Add a new known person to the database
        Can accept either image_path or frame (numpy array)
        """
        try:
            # Create person
            person = self.face_admin_client.large_person_group.create_person(
                large_person_group_id=self.person_group_id,
                name=name
            )
            
            print(f"Created person: {name} (ID: {person.person_id})")
            
            # Add face to person
            if frame is not None:
                # Convert frame to bytes
                _, buffer = cv2.imencode('.jpg', frame)
                image_data = buffer.tobytes()
                
                # Check quality first
                detected_faces = self.face_client.detect(
                    image_content=image_data,
                    detection_model=FaceDetectionModel.DETECTION03,
                    recognition_model=FaceRecognitionModel.RECOGNITION04,
                    return_face_id=True,
                    return_face_attributes=[FaceAttributeTypeRecognition04.QUALITY_FOR_RECOGNITION]
                )
                
                if not detected_faces:
                    return {'success': False, 'error': 'No face detected in image'}
                
                if len(detected_faces) > 1:
                    return {'success': False, 'error': 'Multiple faces detected. Please provide image with single face'}
                
                if detected_faces[0].face_attributes.quality_for_recognition == QualityForRecognition.LOW:
                    return {'success': False, 'error': 'Face quality too low. Please provide clearer image'}
                
                # Add face
                self.face_admin_client.large_person_group.add_face(
                    large_person_group_id=self.person_group_id,
                    person_id=person.person_id,
                    image_content=image_data,
                    detection_model=FaceDetectionModel.DETECTION03
                )
            
            print(f"Face added for person: {name}")
            
            # Train the person group
            print("Training person group...")
            poller = self.face_admin_client.large_person_group.begin_train(
                large_person_group_id=self.person_group_id,
                polling_interval=5
            )
            poller.wait()
            print("Training completed")
            
            return {
                'success': True,
                'person_id': person.person_id,
                'person_name': name
            }
            
        except Exception as e:
            print(f"Error adding person: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def list_known_persons(self):
        """List all known persons in the database"""
        try:
            persons = self.face_admin_client.large_person_group.get_persons(
                large_person_group_id=self.person_group_id
            )
            
            persons_list = []
            for person in persons:
                persons_list.append({
                    'person_id': person.person_id,
                    'name': person.name,
                    'face_count': len(person.persisted_face_ids)
                })
            
            return persons_list
            
        except Exception as e:
            print(f"Error listing persons: {str(e)}")
            return []
    
    def delete_person(self, person_id):
        """Delete a person from the database"""
        try:
            self.face_admin_client.large_person_group.delete_person(
                large_person_group_id=self.person_group_id,
                person_id=person_id
            )
            
            # Retrain after deletion
            poller = self.face_admin_client.large_person_group.begin_train(
                large_person_group_id=self.person_group_id,
                polling_interval=5
            )
            poller.wait()
            
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def analyze_frame_comprehensive(self, frame):
        """
        Comprehensive face analysis including detection, identification, and attributes
        """
        # Detect faces with attributes
        faces_info = self.detect_faces_with_attributes(frame)
        
        if not faces_info:
            return {
                'face_count': 0,
                'faces': [],
                'summary': 'No faces detected'
            }
        
        # Extract face IDs for identification
        face_ids = [face['face_id'] for face in faces_info if face['quality'] != 'LOW']
        
        # Identify faces
        identified_faces = {}
        if face_ids:
            identified_faces = self.identify_faces(face_ids)
        
        # Combine information
        for face in faces_info:
            face_id = face['face_id']
            if face_id in identified_faces:
                face['identification'] = identified_faces[face_id]
            else:
                face['identification'] = {'identified': False, 'reason': 'Quality too low'}
        
        # Generate summary
        known_persons = [f['identification']['person_name'] for f in faces_info 
                        if f['identification'].get('identified')]
        unknown_count = sum(1 for f in faces_info if not f['identification'].get('identified'))
        
        summary_parts = []
        if known_persons:
            summary_parts.append(f"Known: {', '.join(known_persons)}")
        if unknown_count > 0:
            summary_parts.append(f"{unknown_count} unknown person(s)")
        
        summary = '; '.join(summary_parts) if summary_parts else 'Faces detected but not identified'
        
        return {
            'face_count': len(faces_info),
            'faces': faces_info,
            'summary': summary,
            'known_persons': known_persons,
            'unknown_count': unknown_count
        }
