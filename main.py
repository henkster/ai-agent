import argparse
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from call_function import available_functions, call_function
from prompts import system_prompt

def main():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if api_key is None: # if not api_key:
        raise RuntimeError("Gemini API key not set.")

    client = genai.Client(api_key=api_key)

    user_prompt, verbose = get_args()

    messages = [types.Content(role="user", parts=[types.Part(text=user_prompt)])]

    # config=types.GenerateContentConfig(
    #     system_instruction=system_prompt,
    #     temperature=0
    # )

    config=types.GenerateContentConfig(
        tools=[available_functions], system_instruction=system_prompt
    )

    # response = client.models.generate_content(
    #     model="gemini-2.5-flash", contents=messages
    # )

    for _ in range(20):
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=messages,
            config=config,
        )

        if response.candidates:
            for candidate in response.candidates:
                messages.append(candidate.content)

        if response.usage_metadata is None:
            raise RuntimeError("No response usage metadata, possible API request failure.")

        function_results = []
        
        if verbose:
            print(f"User prompt: {user_prompt}")
            print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
            print(f"Response tokens: {response.usage_metadata.candidates_token_count}")
        print("Response:")
        if not response.function_calls is None:
            for function_call in response.function_calls:
                #print(f"Calling function: {function_call.name}({function_call.args})")
                function_call_result = call_function(function_call, verbose)
                if not function_call_result.parts:
                    raise Exception("No parts") #return Error: ...?
                if function_call_result.parts[0].function_response is None:
                    raise Exception("Not function response")
                if function_call_result.parts[0].function_response.response is None:
                    raise Exception("No response")
                function_results.append(function_call_result.parts[0])
                if verbose:
                    print(f"-> {function_call_result.parts[0].function_response.response}")
            messages.append(types.Content(role="user", parts=function_results))
        else:
            print(response.text)
            return
    print("Something went wrong, no results")
    exit(code=1)

def get_args():
    parser = argparse.ArgumentParser(description="AI Code Assistant")
    parser.add_argument("user_prompt", type=str, help="Prompt to send to Gemini")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    # Now we can access `args.user_prompt`
    return args.user_prompt, args.verbose

if __name__ == "__main__":
    main()
