"""
OpenRouter API Connection Test
Tests the OpenRouter API with different models and scenarios
"""

import yaml
import sys
from openai import OpenAI

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

def test_basic_connection(client, model):
    """Test basic API connection"""
    print("\n--- Test 1: Basic Connection ---")
    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://github.com/casc-project",
                "X-Title": "CASC Security Cam Test",
            },
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Reply with exactly: 'API connection successful'"
                }
            ]
        )
        
        response = completion.choices[0].message.content
        print(f"✓ Response received: {response}")
        return True
    except Exception as e:
        print(f"✗ Basic connection failed: {str(e)}")
        return False

def test_system_message(client, model):
    """Test system message functionality"""
    print("\n--- Test 2: System Message ---")
    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://github.com/casc-project",
                "X-Title": "CASC Security Cam Test",
            },
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a security camera AI. Always start responses with 'Security Alert:'"
                },
                {
                    "role": "user",
                    "content": "A person is detected"
                }
            ]
        )
        
        response = completion.choices[0].message.content
        print(f"✓ Response: {response}")
        
        if "Security" in response or "security" in response:
            print("✓ System message working correctly")
            return True
        else:
            print("⚠ System message may not be fully respected")
            return True
    except Exception as e:
        print(f"✗ System message test failed: {str(e)}")
        return False

def test_context_summary(client, model):
    """Test generating context summary (like in real app)"""
    print("\n--- Test 3: Context Summary Generation ---")
    try:
        vision_data = {
            'description': 'a person standing in a room',
            'objects': [{'name': 'person', 'confidence': 0.95}, {'name': 'chair', 'confidence': 0.82}],
            'tags': ['person', 'indoor', 'room', 'standing', 'furniture'],
            'faces': 1
        }
        
        prompt = f"""Based on this security camera analysis, provide a clear, concise summary of what's happening:

Description: {vision_data['description']}
Objects detected: {', '.join([obj['name'] for obj in vision_data['objects']])}
Tags: {', '.join(vision_data['tags'][:5])}
Number of people detected: {vision_data['faces']}

Provide a natural language summary suitable for a security alert."""

        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://github.com/casc-project",
                "X-Title": "CASC Security Cam Test",
            },
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a security camera AI assistant. Provide clear, factual summaries of events."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        response = completion.choices[0].message.content
        print(f"✓ Context summary generated:")
        print(f"  {response}")
        return True
    except Exception as e:
        print(f"✗ Context summary test failed: {str(e)}")
        return False

def test_qa_functionality(client, model):
    """Test Q&A functionality"""
    print("\n--- Test 4: Q&A Functionality ---")
    try:
        event_context = {
            'ai_summary': 'A person was detected standing near the entrance door at 2:30 PM.',
            'vision_analysis': {
                'description': 'a person walking near the entrance',
                'objects': [{'name': 'person', 'confidence': 0.95}, {'name': 'door', 'confidence': 0.87}],
                'tags': ['person', 'indoor', 'door', 'standing'],
                'faces': 1
            },
            'timestamp': '2024-01-15 14:30:00',
            'faces_count': 1,
            'bodies_count': 1,
            'threat_analysis': {
                'alert_level': 'LOW',
                'is_suspicious': False,
                'reason': 'Normal activity detected'
            }
        }
        
        questions = [
            "How many people were detected?",
            "Where was the person located?",
            "What time did this happen?",
            "Was this suspicious?",
            "What objects were detected?"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n  Question {i}: {question}")
            
            # Create detailed context
            context_str = f"""Event Information:
- Summary: {event_context['ai_summary']}
- Timestamp: {event_context['timestamp']}
- Scene: {event_context['vision_analysis']['description']}
- Faces: {event_context['faces_count']}
- People: {event_context['bodies_count']}
- Objects: {', '.join([obj['name'] for obj in event_context['vision_analysis']['objects']])}
- Alert Level: {event_context['threat_analysis']['alert_level']}
- Suspicious: {event_context['threat_analysis']['is_suspicious']}
"""
            
            completion = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://github.com/casc-project",
                    "X-Title": "CASC Security Cam Test",
                },
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a security camera AI. Answer based on this event:\n{context_str}"
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            if completion.choices and len(completion.choices) > 0:
                response = completion.choices[0].message.content
                if response and response.strip():
                    print(f"  Answer: {response}")
                else:
                    print(f"  WARNING: Empty response received")
                    return False
            else:
                print(f"  ERROR: No choices in response")
                return False
        
        print("\n✓ Q&A functionality working")
        return True
    except Exception as e:
        print(f"✗ Q&A test failed: {str(e)}")
        return False

def test_model_availability(client, models_to_test):
    """Test multiple free models"""
    print("\n--- Test 5: Model Availability ---")
    working_models = []
    
    for model in models_to_test:
        print(f"\n  Testing model: {model}")
        try:
            completion = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://github.com/casc-project",
                    "X-Title": "CASC Security Cam Test",
                },
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": "Say 'OK'"
                    }
                ]
            )
            print(f"  ✓ {model} - WORKING")
            working_models.append(model)
        except Exception as e:
            print(f"  ✗ {model} - NOT AVAILABLE: {str(e)}")
    
    return working_models

def main():
    print("="*70)
    print("OpenRouter API Connection Tester")
    print("="*70)
    
    # Load config
    config = load_config()
    if not config:
        sys.exit(1)
    
    api_key = config['openrouter']['api_key']
    model = config['openrouter']['model']
    
    if not api_key or api_key == "YOUR_OPENROUTER_API_KEY":
        print("\n✗ OpenRouter API key not configured!")
        print("  Please add your API key to config/config.yaml")
        sys.exit(1)
    
    print(f"\nConfigured model: {model}")
    print(f"API key: {api_key[:10]}...{api_key[-4:]}")
    
    # Create client
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        print("✓ OpenRouter client created")
    except Exception as e:
        print(f"✗ Failed to create client: {str(e)}")
        sys.exit(1)
    
    # Run tests
    results = {}
    results['basic_connection'] = test_basic_connection(client, model)
    results['system_message'] = test_system_message(client, model)
    results['context_summary'] = test_context_summary(client, model)
    results['qa_functionality'] = test_qa_functionality(client, model)
    
    # Test alternative models
    free_models = [
        "deepseek/deepseek-chat-v3.1:free",
        "mistralai/mistral-7b-instruct:free",
        "google/gemini-2.0-flash-lite:free",
        "qwen/qwen-2.5-7b-instruct:free"
    ]
    
    print("\n" + "="*70)
    working_models = test_model_availability(client, free_models)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nWorking Models: {len(working_models)}/{len(free_models)}")
    if working_models:
        print("Available models:")
        for model in working_models:
            print(f"  - {model}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL OPENROUTER TESTS PASSED")
        print("  OpenRouter integration is ready!")
    else:
        print("✗ SOME TESTS FAILED")
        print("  Check API key and model availability")
    print("="*70)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
