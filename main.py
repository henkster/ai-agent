import argparse
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from prompts import *

def main():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if api_key is None: # if not api_key:
        raise RuntimeError("Gemini API key not set.")

    client = genai.Client(api_key=api_key)

    user_prompt, verbose = get_args()

    messages = [types.Content(role="user", parts=[types.Part(text=user_prompt)])]

    # response = client.models.generate_content(
    #     model="gemini-2.5-flash", contents=messages
    # )

    model_name = "gemini-2.5-flash"

    config=types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0
    )

    response = client.models.generate_content(
        model=model_name,
        contents=messages,
        config=config,
    )

    if response.usage_metadata is None:
        raise RuntimeError("No response usage metadata, possible API request failure.")
    
    if verbose:
        print(f"User prompt: {user_prompt}")
        print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
        print(f"Response tokens: {response.usage_metadata.candidates_token_count}")
    print("Response:")
    print(response.text)

def get_args():
    parser = argparse.ArgumentParser(description="AI Code Assistant")
    parser.add_argument("user_prompt", type=str, help="Prompt to send to Gemini")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    # Now we can access `args.user_prompt`
    return args.user_prompt, args.verbose

if __name__ == "__main__":
    main()
