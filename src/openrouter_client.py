import json
from openai import OpenAI

class OpenRouterClient:
    def __init__(self, config):
        self.api_key = config['openrouter']['api_key']
        self.model = config['openrouter']['model']
        self.conversation_history = []
        
        # Initialize OpenAI client with OpenRouter base URL
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key
        )
    
    def generate_context_summary(self, vision_analysis):
        """Generate human-readable summary from Azure Vision analysis"""
        prompt = f"""Based on this security camera analysis, provide a clear, concise summary of what's happening:

Description: {vision_analysis['description']}
Objects detected: {', '.join([obj['name'] for obj in vision_analysis['objects']])}
Tags: {', '.join(vision_analysis['tags'][:5])}
Number of people detected: {vision_analysis['faces']}

Provide a natural language summary suitable for a security alert."""

        response = self._call_api(prompt, system_message="You are a security camera AI assistant. Provide clear, factual summaries of events.")
        return response
    
    def answer_question(self, question, event_context):
        """Answer user questions about a security event"""
        # Create a detailed context string
        context_str = f"""Event Information:
- Summary: {event_context.get('ai_summary', 'No summary available')}
- Timestamp: {event_context.get('timestamp', 'Unknown')}
- Event ID: {event_context.get('event_id', 'Unknown')}
"""
        
        if event_context.get('vision_analysis'):
            vision = event_context['vision_analysis']
            context_str += f"""
- Scene Description: {vision.get('description', 'N/A')}
- Detected Objects: {', '.join([obj['name'] for obj in vision.get('objects', [])])}
- Tags: {', '.join(vision.get('tags', [])[:5])}
- Number of Faces: {vision.get('faces', 0)}
"""
        
        if 'faces_count' in event_context:
            context_str += f"\n- Faces Detected: {event_context['faces_count']}"
        
        if 'bodies_count' in event_context:
            context_str += f"\n- People Detected: {event_context['bodies_count']}"
        
        if 'threat_analysis' in event_context:
            threat = event_context['threat_analysis']
            context_str += f"""
- Alert Level: {threat.get('alert_level', 'Unknown')}
- Suspicious: {threat.get('is_suspicious', False)}
- Reason: {threat.get('reason', 'N/A')}
"""
        
        system_message = f"""You are a security camera AI assistant. Answer the user's question based on this security event information.

{context_str}

Provide clear, specific answers based on the information above. If you don't have enough information to answer, say so."""

        response = self._call_api(question, system_message=system_message)
        
        # Store in conversation history
        self.conversation_history.append({
            'question': question,
            'answer': response
        })
        
        return response
    
    def _call_api(self, user_message, system_message=None):
        """Call OpenRouter API using OpenAI client"""
        try:
            messages = []
            
            if system_message:
                messages.append({
                    "role": "system",
                    "content": system_message
                })
            
            messages.append({
                "role": "user",
                "content": user_message
            })
            
            # Add timeout and better parameters
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://github.com/casc-project",
                    "X-Title": "CASC Security Cam",
                },
                model=self.model,
                messages=messages,
                max_tokens=800,  # Increased from 500
                temperature=0.7,
                top_p=0.9
            )
            
            # Extract response
            if completion.choices and len(completion.choices) > 0:
                response = completion.choices[0].message.content
                
                # Check if response is empty or None
                if not response or response.strip() == "":
                    print(f"DEBUG: Empty response from model {self.model}")
                    return self._generate_fallback_response(user_message, system_message)
                
                return response.strip()
            else:
                print(f"DEBUG: No choices in response from {self.model}")
                return self._generate_fallback_response(user_message, system_message)
                
        except Exception as e:
            error_msg = str(e)
            print(f"DEBUG: OpenRouter API error: {error_msg}")
            
            # Provide more helpful error messages
            if "401" in error_msg or "authentication" in error_msg.lower():
                return "API authentication error. Please check your OpenRouter API key in config.yaml"
            elif "429" in error_msg or "rate limit" in error_msg.lower():
                return "Rate limit exceeded. Free tier has limited requests. Please wait a moment."
            elif "timeout" in error_msg.lower():
                return "Request timed out. Please try again."
            elif "credit" in error_msg.lower() or "insufficient" in error_msg.lower():
                return "API credits exhausted. Please check your OpenRouter account."
            else:
                return self._generate_fallback_response(user_message, system_message)
    
    def _generate_fallback_response(self, user_message, system_message):
        """Generate a basic fallback response when API fails"""
        # Try to extract context from system message
        context_info = ""
        if system_message and "Event Information:" in system_message:
            # Extract key info from context
            lines = system_message.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in ['summary:', 'description:', 'person:', 'emotion:', 'age:', 'gender:']):
                    context_info += line + "\n"
        
        if context_info:
            return f"Based on the available information:\n{context_info}\n\nNote: Full AI analysis unavailable. This is basic information from the event."
        else:
            return "I'm unable to process your question right now. The AI service is experiencing issues. Please try:\n1. Rephrasing your question\n2. Checking the event details with 'show' command\n3. Trying again in a few moments"

    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
