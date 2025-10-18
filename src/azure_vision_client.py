from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
import cv2
from io import BytesIO

class AzureVisionClient:
    def __init__(self, config):
        self.endpoint = config['azure_vision']['endpoint']
        self.api_key = config['azure_vision']['api_key']
        
        credentials = CognitiveServicesCredentials(self.api_key)
        self.client = ComputerVisionClient(self.endpoint, credentials)
    
    def analyze_image(self, frame):
        """
        Analyze image using Azure Computer Vision
        Returns detailed description of what's in the image
        """
        try:
            # Convert frame to bytes
            _, buffer = cv2.imencode('.jpg', frame)
            image_stream = BytesIO(buffer.tobytes())
            
            # Analyze image
            features = [
                VisualFeatureTypes.description,
                VisualFeatureTypes.objects,
                VisualFeatureTypes.tags,
                VisualFeatureTypes.faces
            ]
            
            analysis = self.client.analyze_image_in_stream(
                image_stream, 
                visual_features=features
            )
            
            # Extract relevant information
            result = {
                'description': analysis.description.captions[0].text if analysis.description.captions else "No description available",
                'confidence': analysis.description.captions[0].confidence if analysis.description.captions else 0,
                'tags': [tag.name for tag in analysis.tags],
                'objects': [{'name': obj.object_property, 'confidence': obj.confidence} for obj in analysis.objects],
                'faces': len(analysis.faces) if analysis.faces else 0
            }
            
            return result
            
        except Exception as e:
            print(f"Error analyzing image: {str(e)}")
            return None
