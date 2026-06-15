import boto3
import json

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful assistant. You can look up weather, find schools in the USA, "
    "and provide cooking recipes. Use the available tools whenever the user's question "
    "calls for one of those topics."
)

MAX_TURNS  = 2          # keep last N user/assistant turn pairs in history
MAX_TOKENS = 200     # max tokens in Claude's reply
MODEL_ID   = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

# ─────────────────────────────────────────────
# TOOL DEFINITIONS  (sent to Bedrock every call)
# ─────────────────────────────────────────────

TOOLS = [
    {
        "name": "getweather",
        "description": (
            "Returns the current weather for a given city. "
            "Use this whenever the user asks about weather or temperature."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 'Chicago'."}
            },
            "required": ["city"]
        }
    },
    {
        "name": "getschoolname",
        "description": (
            "Returns the names of schools or universities in a given US city or zip code. "
            "Use this when the user asks about schools, colleges, or educational institutions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name or zip code."}
            },
            "required": ["location"]
        }
    },
    {
        "name": "getcookingrecipe",
        "description": (
            "Returns a simple cooking recipe for a requested dish. "
            "Use this when the user asks how to cook or make a specific food."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dish": {"type": "string", "description": "Dish name, e.g. 'pasta carbonara'."}
            },
            "required": ["dish"]
        }
    }
]

# ─────────────────────────────────────────────
# MOCK TOOL IMPLEMENTATIONS
# ─────────────────────────────────────────────

def getweather(city: str) -> dict:
    data = {
        "chicago":     {"temperature": "72°F", "condition": "Partly Cloudy", "humidity": "55%"},
        "new york":    {"temperature": "68°F", "condition": "Sunny",          "humidity": "50%"},
        "los angeles": {"temperature": "85°F", "condition": "Clear",          "humidity": "35%"},
        "naperville":  {"temperature": "70°F", "condition": "Mostly Sunny",   "humidity": "52%"},
    }
    return data.get(city.lower(), {"temperature": "75°F", "condition": "Fair", "humidity": "45%"})


def getschoolname(location: str) -> dict:
    data = {
        "naperville": {"schools": ["Naperville North High School",
                                   "Naperville Central High School",
                                   "North Central College"]},
        "chicago":    {"schools": ["University of Chicago",
                                   "DePaul University",
                                   "Northwestern University (Evanston)"]},
        "60540":      {"schools": ["Naperville North High School",
                                   "Jefferson Junior High School"]},
    }
    return data.get(
        location.lower(),
        {"schools": [f"Lincoln Elementary in {location}",
                     f"{location} Community College"]}
    )


def getcookingrecipe(dish: str) -> dict:
    data = {
        "pasta carbonara": {
            "ingredients": ["200g spaghetti", "100g pancetta", "2 eggs",
                            "50g Pecorino Romano", "Black pepper", "Salt"],
            "steps": ["Cook pasta in salted water.",
                      "Fry pancetta until crispy.",
                      "Whisk eggs with Pecorino and pepper.",
                      "Toss hot pasta with pancetta off the heat.",
                      "Pour egg mixture over, add pasta water, toss quickly.",
                      "Serve immediately."],
            "prep_time": "20 minutes", "servings": 2
        },
        "chocolate chip cookies": {
            "ingredients": ["2¼ cups flour", "1 tsp baking soda", "1 cup butter",
                            "¾ cup each sugar & brown sugar", "2 eggs",
                            "2 tsp vanilla", "2 cups chocolate chips"],
            "steps": ["Preheat oven to 375°F.",
                      "Cream butter and sugars; beat in eggs and vanilla.",
                      "Blend in flour and baking soda.",
                      "Stir in chocolate chips.",
                      "Drop spoonfuls on baking sheet; bake 9-11 min."],
            "prep_time": "25 minutes", "servings": 48
        }
    }
    return data.get(
        dish.lower(),
        {
            "ingredients": [f"Main ingredient for {dish}", "Salt", "Pepper", "Olive oil"],
            "steps": ["Prepare ingredients.", "Cook to your liking.", "Season and serve."],
            "prep_time": "30 minutes", "servings": 4
        }
    )


TOOL_FUNCTIONS = {
    "getweather":       getweather,
    "getschoolname":    getschoolname,
    "getcookingrecipe": getcookingrecipe,
}

def execute_tool(name: str, tool_input: dict) -> str:
    """Dispatch to the right mock function and return JSON string."""
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    return json.dumps(fn(**tool_input))

# ─────────────────────────────────────────────
# BEDROCK HELPERS
# ─────────────────────────────────────────────

client = boto3.client("bedrock-runtime", region_name="us-east-1")

def invoke_model(messages: list) -> dict:
    """
    Send the full conversation (with tools) to Bedrock and return the parsed body.
    """
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS,
        "system":     SYSTEM_PROMPT,
        "tools":      TOOLS,          # always include tool definitions
        "messages":   messages
    }
    response = client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(body)         # convert Python dict to JSON string
    )
    return json.loads(response["body"].read())


def extract_text(response_body: dict) -> str:
    """Pull all text blocks out of a Bedrock response and join them."""
    return " ".join(
        b.get("text", "")
        for b in response_body.get("content", [])
        if b.get("type") == "text"
    ).strip()


def extract_tool_use(response_body: dict):
    """Return (tool_use_id, tool_name, tool_input) if the model called a tool."""
    for b in response_body.get("content", []):
        if b.get("type") == "tool_use":
            return b["id"], b["name"], b["input"]
    return None, None, None

# ─────────────────────────────────────────────
# CORE: ONE USER TURN  (may involve a tool call)
# ─────────────────────────────────────────────

def process_turn(messages: list) -> tuple[str, list]:
    """
    Send messages to Bedrock.  If the model picks a tool:
      1. Execute it locally.
      2. Append the tool result to messages.
      3. Call Bedrock again for the final answer.
    Returns (reply_text, updated_messages).
    """
    # ── Step 1: initial call ──────────────────────────────────
    result1      = invoke_model(messages)
    stop_reason  = result1.get("stop_reason")
    usage        = result1.get("usage", {})
    print(f"  [tokens step-1 — in: {usage.get('input_tokens','?')}, "
          f"out: {usage.get('output_tokens','?')}]")

    tool_id, tool_name, tool_input = extract_tool_use(result1)

    # No tool needed — return the text directly
    if stop_reason != "tool_use" or tool_name is None:
        reply = extract_text(result1)
        messages.append({"role": "assistant", "content": result1["content"]})
        return reply, messages

    # ── Step 2: run the tool ──────────────────────────────────
    print(f"  [tool call → {tool_name}({tool_input})]")
    tool_result = execute_tool(tool_name, tool_input)
    print(f"  [tool result → {tool_result}]")

    # Append assistant's tool-use turn, then our tool result
    messages.append({"role": "assistant", "content": result1["content"]})
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": tool_result,
            }
        ]
    })

    # ── Step 3: second call with tool result ──────────────────
    result2 = invoke_model(messages)
    usage2  = result2.get("usage", {})
    print(f"  [tokens step-2 — in: {usage2.get('input_tokens','?')}, "
          f"out: {usage2.get('output_tokens','?')}]")

    reply = extract_text(result2)
    messages.append({"role": "assistant", "content": result2["content"]})
    return reply, messages

# ─────────────────────────────────────────────
# MAIN: CONVERSATION LOOP
# ─────────────────────────────────────────────

def is_tool_result_message(msg: dict) -> bool:
    """Return True if this message is a tool_result user-turn (not a real user message)."""
    content = msg.get("content", [])
    return (
        msg.get("role") == "user"
        and isinstance(content, list)
        and len(content) > 0
        and content[0].get("type") == "tool_result"
    )


def trim_history(messages: list, max_turns: int) -> list:
    """
    Keep only the last max_turns COMPLETE logical turns in history.

    A logical turn is:
      [user text]  →  [assistant tool_use]  →  [user tool_result]  →  [assistant text]
    or simply:
      [user text]  →  [assistant text]

    The root cause of the ValidationException is trimming the assistant
    tool_use message while leaving the paired tool_result message — Bedrock
    then sees a tool_result with no matching tool_use and rejects the request.

    Strategy:
      1. Walk the list backwards grouping messages into logical turns.
         A new logical turn starts at every "real" user message
         (i.e. role==user AND content is a plain string or non-tool_result list).
      2. Collect the last max_turns groups.
      3. Never cut mid-group — always keep the full tool_use+tool_result pair.
    """
    if not messages:
        return messages

    # Split into logical turns by finding "real" user message boundaries.
    # Each group: [user_msg, asst_tool_use?, user_tool_result?, asst_final]
    groups: list[list[dict]] = []
    current_group: list[dict] = []

    for msg in messages:
        if msg.get("role") == "user" and not is_tool_result_message(msg):
            # Start of a new logical turn — save the previous group first
            if current_group:
                groups.append(current_group)
            current_group = [msg]
        else:
            # assistant message OR tool_result user message — part of current turn
            current_group.append(msg)

    if current_group:
        groups.append(current_group)

    # Keep only the last max_turns complete groups
    trimmed_groups = groups[-max_turns:]
    return [msg for group in trimmed_groups for msg in group]


if __name__ == "__main__":
    messages = []
    print("\nStart chatting! Type 'quit' to exit.\n")
    print("Try: 'weather in Chicago', 'schools in Naperville', 'how to make cookies'\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break

        # Trim old history BEFORE adding the new user message
        messages = trim_history(messages, MAX_TURNS)

        # Add the new user message
        messages.append({"role": "user", "content": user_input})

        # Process the turn (runs tool if needed, returns final text)
        reply, messages = process_turn(messages)

        print(f"\nClaude: {reply}\n")