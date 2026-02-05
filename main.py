import argparse
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

def main():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if api_key is None: # if not api_key:
        raise RuntimeError("Gemini API key not set.")

    client = genai.Client(api_key=api_key)

    user_prompt = get_prompt()

    messages = [types.Content(role="user", parts=[types.Part(text=user_prompt)])]

    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=messages
    )

    if response.usage_metadata is None:
        raise RuntimeError("No response usage metadata, possible API request failure.")
    
    print(f"User prompt: {user_prompt}")
    print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
    print(f"Response tokens: {response.usage_metadata.candidates_token_count}")
    print("Response:")
    print(response.text)

def get_prompt():
    parser = argparse.ArgumentParser(description="AI Code Assistant")
    parser.add_argument("user_prompt", type=str, help="Prompt to send to Gemini")
    args = parser.parse_args()
    # Now we can access `args.user_prompt`
    return args.user_prompt

if __name__ == "__main__":
    main()
