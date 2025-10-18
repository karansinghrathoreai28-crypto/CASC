"""
Quick diagnostic test for OpenRouter API issues
"""

import yaml
from openai import OpenAI

def test_openrouter():
    print("="*70)
    print("OpenRouter Diagnostic Test")
    print("="*70)
    
    # Load config
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    api_key = config['openrouter']['api_key']
    model = config['openrouter']['model']
    
    print(f"\nAPI Key: {api_key[:20]}...{api_key[-10:]}")
    print(f"Model: {model}")
    
    # Test models
    test_models = [
        
        "mistralai/mistral-7b-instruct:free"
    ]
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )
    
    for test_model in test_models:
        print(f"\n--- Testing: {test_model} ---")
        try:
            completion = client.chat.completions.create(
                model=test_model,
                messages=[
                    {"role": "user", "content": "Say 'Hello, I am working!' in one sentence."}
                ],
                max_tokens=100,
                temperature=0.7
            )
            
            if completion.choices:
                response = completion.choices[0].message.content
                print(f"✓ SUCCESS: {response}")
            else:
                print(f"✗ FAILED: No response")
                
        except Exception as e:
            print(f"✗ ERROR: {str(e)}")
    
    print("\n" + "="*70)
    print("RECOMMENDATION:")
    print("Update config.yaml with a working model from above")
    print("="*70)

if __name__ == "__main__":
    test_openrouter()
