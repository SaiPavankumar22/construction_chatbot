---
title: Construction Company AI Assistant
emoji: 🏗️
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: "5.29.1"
app_file: app.py
pinned: false
---
# Construction Company AI Assistant
An expert, construction-only AI assistant with a modern Gradio UI, powered by DeepSeek R1 via OpenRouter, optional live web search (Serper), and CrewAI agents.

## Features

- **Construction-only assistant**: Politely declines non-construction queries
- **DeepSeek R1 reasoning model**: Via OpenRouter
- **Optional live web search**: Uses Serper to fetch current prices, codes, and updates
- **Memory**: Remembers the last 5 Q&A pairs
- **CrewAI agents**: Expert assistant + optional research agent
- **Modern UI**: Construction-themed styling, examples, and status indicators

## What’s inside (tech)

- **Model**: DeepSeek R1 (OpenRouter)
- **Agent framework**: CrewAI (`Agent`, `Task`, `Crew`, `LLM`)
- **Web search tool**: `SerperDevTool` (optional)
- **UI**: Gradio (messages format), auto-launches browser

## Installation

1) Create/activate a Python 3.10+ environment.

2) Install dependencies:
```bash
pip install gradio crewai crewai-tools langchain-openai
```

## Configuration

The app is preconfigured in `app.py` to use:
- OpenRouter (DeepSeek R1) for responses
- Optional Serper for web search (set `SERPER_API_KEY` to enable)

Set Serper (optional) if you want live web search:
- Windows (PowerShell):
```powershell
$env:SERPER_API_KEY="your-serper-api-key"
```
- Linux/Mac:
```bash
export SERPER_API_KEY="your-serper-api-key"
```

If you prefer, you can also run the optional setup helper:
```bash
python setup_env.py
```

## Run

```bash
python app.py
```

By default the app launches on:
- Local: `http://localhost:7863`
- A public Gradio share link (shown in the console)

The UI auto-opens in your browser (`inbrowser=True`).

## How it works (high-level)

- Incoming questions are first filtered to ensure they are construction-related.
- For time-sensitive questions (e.g., “current”, “latest”, “2024/2025”, “price”, “cost”, “regulation”), a research agent may search the web (if Serper is enabled) and pass findings to the expert agent.
- If anything fails, a direct LLM fallback path still answers construction questions.

## Example questions

- “Latest fire safety codes for high-rise buildings (2024)?”
- “Current price of M30 ready-mix concrete per m³ in my area?”
- “OSHA requirements for scaffolding on construction sites?”
- “How to size rebar for a medium-span concrete beam?”
- “Best project management approach for a multi-tower site?”

## Troubleshooting

- **Port already in use**: Change `server_port` in `app.py` (default is 7863).
- **Web search disabled**: Set `SERPER_API_KEY` and restart.
- **Import errors**: Run the install command again and ensure your environment is active.

## Notes

- The assistant strictly answers construction-related topics (building, safety, materials, project management, engineering). Non-construction queries are declined by design.

