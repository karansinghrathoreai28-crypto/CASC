from azure.cosmos import CosmosClient, PartitionKey, exceptions
from datetime import datetime
import uuid
import base64
import cv2

class DatabaseManager:
    def __init__(self, config):
        self.config = config
        self.client = CosmosClient(
            config['cosmos_db']['endpoint'],
            config['cosmos_db']['key']
        )
        
        self.database_name = config['cosmos_db']['database_name']
        self.container_name = config['cosmos_db']['container_name']
        
        self._initialize_database()
    
    def _initialize_database(self):
        """Create database and container if they don't exist"""
        try:
            self.database = self.client.create_database_if_not_exists(id=self.database_name)
            self.container = self.database.create_container_if_not_exists(
                id=self.container_name,
                partition_key=PartitionKey(path="/event_id")
                # Removed offer_throughput - not supported for serverless accounts
            )
        except exceptions.CosmosHttpResponseError as e:
            print(f"Error initializing database: {str(e)}")
    
    def save_event(self, frame, vision_analysis, ai_summary, motion_level):
        """Save security event to database"""
        event_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        # Encode image as base64 if configured to save
        image_data = None
        if self.config['storage']['save_images']:
            _, buffer = cv2.imencode('.jpg', frame)
            image_data = base64.b64encode(buffer).decode('utf-8')
        
        event_document = {
            'id': event_id,
            'event_id': event_id,
            'timestamp': timestamp,
            'motion_level': motion_level,
            'vision_analysis': vision_analysis,
            'ai_summary': ai_summary,
            'image_data': image_data,
            'conversations': [],
            'type': 'security_event'
        }
        
        try:
            self.container.create_item(body=event_document)
            return event_id
        except exceptions.CosmosHttpResponseError as e:
            print(f"Error saving event: {str(e)}")
            return None
    
    def add_conversation(self, event_id, question, answer):
        """Add Q&A to an event"""
        try:
            event = self.container.read_item(
                item=event_id,
                partition_key=event_id
            )
            
            conversation_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'question': question,
                'answer': answer
            }
            
            event['conversations'].append(conversation_entry)
            
            self.container.replace_item(
                item=event_id,
                body=event
            )
            return True
        except exceptions.CosmosHttpResponseError as e:
            print(f"Error adding conversation: {str(e)}")
            return False
    
    def get_event(self, event_id):
        """Retrieve event by ID"""
        try:
            return self.container.read_item(
                item=event_id,
                partition_key=event_id
            )
        except exceptions.CosmosHttpResponseError as e:
            print(f"Error retrieving event: {str(e)}")
            return None
    
    def get_recent_events(self, limit=10):
        """Get recent security events"""
        query = f"SELECT * FROM c WHERE c.type = 'security_event' ORDER BY c.timestamp DESC OFFSET 0 LIMIT {limit}"
        
        try:
            items = list(self.container.query_items(
                query=query,
                enable_cross_partition_query=True
            ))
            return items
        except exceptions.CosmosHttpResponseError as e:
            print(f"Error querying events: {str(e)}")
            return []
