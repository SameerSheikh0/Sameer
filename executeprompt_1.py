import boto3
import json

from huggingface_hub import User
from streamlit import user

def call_claude(system_prompt: str, user_prompt: str, model_id: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0") -> str:
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    
    msg1 = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 200,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_prompt}
        ]
    }
     
    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(msg1)
    )

    result = json.loads(response["body"].read())
    return result["content"][0]["text"]


if __name__ == "__main__":
    #system = input("System prompt: ")
    #user = input("User prompt: ")
    system = "You are a helpful assistant."
    user="give me school that teachs computer science"
    
    print("\nResponse:")
    print(call_claude(system, user))
