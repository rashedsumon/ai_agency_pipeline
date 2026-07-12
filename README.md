# AI Video Production Pipeline Automation Agent

A production-grade GenAI tool built for transforming raw topical summaries (such as medical text regarding Metformin long-term side effects) into highly structured, multi-modal slide-deck script tables optimized for YouTube Shorts, Reels, and TikTok.

## Core Architecture
- **LangGraph**: Orchestrates a multi-step agent workflow (Consolidation -> Visual Prompting -> Bulk Creation Formatting).
- **LangChain**: Abstracts the underlying LLM interfaces and structures state management.
- **Hugging Face Hub / Transformers**: Scaffolding for downloading task-specific datasets and performing downstream Parameter-Efficient Fine-Tuning (PEFT).
- **Streamlit**: Front-end interface for agency operators to process raw inputs in bulk.

## Getting Started

1. Clone this repository to your Streamlit Cloud environment or local machine.
2. Ensure you have your `OPENAI_API_KEY` or appropriate Model API keys configured in Streamlit Secrets or your local environment.
3. Run the interface locally:
   ```bash
   pip install -r requirements.txt
   streamlit run streamlit_app.py