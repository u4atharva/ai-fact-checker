# Autonomous Wikipedia Fact-Checker Agent

An Agentic AI application designed to combat "AI Hallucinations" by strictly grounding its answers against a real source of truth (Wikipedia). Built using the natively supported `google-genai` SDK and the free `wikipedia` Python package.

## How It Works
Large Language Models are prone to making up facts. In this project, we prompt the AI to take on a strict "Fact-Checker" persona and give it the `search_wikipedia` tool. 

When you provide a claim:
1. The AI extracts the core entities it needs to research.
2. It pauses its text generation to call the `wikipedia` tool.
3. Our Python script queries Wikipedia, summarizes the relevant page, and feeds the encyclopedia data back to the LLM.
4. The AI outputs a final `[TRUE]`, `[FALSE]`, or `[PARTIALLY TRUE]` verdict backed directly by the cited data.

## Setup Instructions

1. Clone this repository to your local machine.
2. Create python virtual environment and install packages:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Copy your Gemini API key to a `.env` file:
   ```bash
   echo "GEMINI_API_KEY=your_actual_key" > .env
   ```

## Usage
Run the script natively and follow the interactive prompts!

```bash
python fact_checker.py
```
Or append your claim directly:
```bash
python fact_checker.py "Did Abraham Lincoln invent the telephone?"
```

## Why this is important
This project demonstrates **RAG (Retrieval-Augmented Generation)** techniques without requiring vector databases or complex embedding pipelines. It shows how simple tool-calling can constrain an LLM to rely on external facts rather than internal weights.
