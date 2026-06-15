import boto3
import json

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# Hardcoded system prompt — tells Claude who it is and what it should do
SYSTEM_PROMPT = "You are a helpful assistant to provide me finding academic schools in usa with ranks."

# Only keep the last N conversation turns to minimize input tokens
# Each turn = 1 user message + 1 assistant message = 2 messages
MAX_TURNS = 2

# Maximum number of tokens Claude can use in its response
# Lower = cheaper, but Claude may cut off long answers
MAX_TOKENS = 200

# The Claude model to use — Haiku is the cheapest and fastest
MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


# ─────────────────────────────────────────────
# FUNCTION: CALL CLAUDE
# ─────────────────────────────────────────────

def call_claude(messages: list) -> str:
    """
    Sends the conversation history to Claude and returns its reply.
    
    :param messages: List of {"role": "user"/"assistant", "content": "..."} dicts
    :return: Claude's reply as a plain string
    """

    # Create the Bedrock client — this connects to AWS Bedrock in us-east-1
    client = boto3.client("bedrock-runtime", region_name="us-east-1")

    # Build the request body that Bedrock expects
    body = {
        "anthropic_version": "bedrock-2023-05-31",  # required by Bedrock, always this value
        "max_tokens": MAX_TOKENS,                    # max tokens Claude can reply with
        "system": SYSTEM_PROMPT,                     # hardcoded system prompt
        "messages": messages                         # full conversation array (trimmed history)
    }

    # Send the request to Bedrock and get a response
    response = client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(body)   # convert Python dict to JSON string
    )

    # Parse the response body from JSON bytes into a Python dict
    # response["body"] is a streaming object, so we call .read() to get the bytes
    result = json.loads(response["body"].read())

    # Print token usage so we can track cost
    # input_tokens  = tokens sent (system prompt + all messages)
    # output_tokens = tokens in Claude's reply
    usage = result["usage"]
    print(f"[Tokens — input: {usage['input_tokens']}, output: {usage['output_tokens']}]\n")

    # Extract and return just the text from Claude's reply
    # result["content"] is an array, we take the first item's "text" field
    return result["content"][0]["text"]


# ─────────────────────────────────────────────
# MAIN: CONVERSATION LOOP
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # Start with an empty message history
    messages = []

    print("\nStart chatting! Type 'quit' to exit.\n")

    while True:

        # Get input from the user
        user_input = input("You: ")

        # Exit the loop if user types 'quit'
        if user_input.lower() == "quit":
            break

        # ── Trim history to save tokens ──────────────────
        # Before adding the new message, trim the history to only
        # keep the last MAX_TURNS turns (MAX_TURNS * 2 messages)
        # e.g. MAX_TURNS=2 keeps last 4 messages (2 user + 2 assistant)
        # This prevents the input token count from growing too large
        messages = messages[-(MAX_TURNS * 2):]

        # Add the new user message to the history
        messages.append({"role": "user", "content": user_input})

        # Send the full (trimmed) history to Claude and get a reply
        reply = call_claude(messages)

        # Print Claude's reply
        print(f"Claude: {reply}\n")

        # Add Claude's reply to the history so the next turn has context
        # We only store the text — not the full result object — to save tokens
        messages.append({"role": "assistant", "content": reply})