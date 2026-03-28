import os
import sys
import wikipedia
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

if not os.environ.get("GEMINI_API_KEY"):
    print("\n[ERROR] Missing GEMINI_API_KEY!")
    sys.exit(1)

client = genai.Client()

# ==========================================
# 1. DEFINE OUR FACT CHECKING TOOLS
# ==========================================

def search_wikipedia(query: str) -> str:
    """Searches Wikipedia and returns a summary of the most relevant article findings. Use this whenever you must verify facts, dates, people, or events."""
    print(f"\n[🔍 FACT CHECKING] Searching Wikipedia for: '{query}'...")
    try:
        # We fetch the top search results to ensure we have the right page title
        search_results = wikipedia.search(query, results=1)
        if not search_results:
            return "No Wikipedia page found for this query."
        
        best_match = search_results[0]
        # Get the summary of the page (limit to 3 sentences to stay concise)
        summary = wikipedia.summary(best_match, sentences=4)
        return f"WIKIPEDIA ARTICLE: {best_match}\nSUMMARY: {summary}"
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Query was too ambiguous. Please be more specific. Possible options include: {e.options[:5]}"
    except wikipedia.exceptions.PageError:
        return "Page not found."
    except Exception as e:
        return f"Wikipedia Error: {e}"

# The agent's abilities
MY_TOOLS = [search_wikipedia]

# ==========================================
# 2. DEFINING THE AGENT LOOP
# ==========================================

def run_agent_loop(user_claim: str):
    print(f"\n🗣️  THE CLAIM: '{user_claim}'\n")
    print("Agent is verifying the claim... Please wait.")
    print("-" * 50)
    
    # We write a strong "System Prompt" alongside the user prompt to give the AI a persona.
    system_instructions = (
        "You are a strict, objective fact-checker. A user will provide a claim. "
        "You MUST use the Wikipedia tool to verify their facts before giving any answer. "
        "Do not rely solely on your internal knowledge. "
        "Your final answer should be formatted starting with one of: [✅ TRUE], [❌ FALSE], or [⚠️ PARTIALLY TRUE]. "
        "Follow this with a brief explanation and your cited Wikipedia sources."
    )
    
    # Send the first message with tools enabled
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=f"System Instruction: {system_instructions}\n\nPlease fact check this user claim: {user_claim}",
        config=types.GenerateContentConfig(
            tools=MY_TOOLS,
            temperature=0.0, # 0.0 temperature for strictly factual outputs
        ),
    )
    
    # The Loop
    while True:
        if response.function_calls:
            tool_responses = []
            
            for tool_call in response.function_calls:
                function_name = tool_call.name
                arguments = tool_call.args
                
                result = ""
                if function_name == "search_wikipedia":
                    result = search_wikipedia(query=arguments.get('query', ''))
                else:
                    result = f"Error: Tool '{function_name}' not found."
                
                tool_responses.append(
                    types.Part.from_function_response(
                        name=function_name,
                        responses={"result": result}
                    )
                )

            # Feed tool results back to the model
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=tool_responses,
                config=types.GenerateContentConfig(tools=MY_TOOLS),
            )
            
        else:
            # We got our final answer!
            print("\n🤖 VERDICT:")
            print("===================================\n")
            print(response.text)
            break
            

if __name__ == "__main__":
    print("\n=== THE AUTONOMOUS FACT-CHECKER ===")
    
    if len(sys.argv) > 1:
        claim = " ".join(sys.argv[1:])
    else:
        claim = input("\nWhat claim would you like me to fact-check? (e.g., 'Abraham Lincoln invented the telephone'):\n> ")
    
    run_agent_loop(claim)
