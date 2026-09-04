# Rwanda Agricultural Statistics RAG Pipeline

A Retrieval-Augmented Generation (RAG) system for querying official Rwandan agricultural survey data and statistical reports using vector search and large language model inference.

## Overview

This application indexes official agricultural datasets (including MINAGRI livestock reports, NAEB coffee and tea statistics, and NISR survey data) into a local vector database. Users can submit natural language queries to retrieve context-specific metrics, yields, export volumes, and policy subsidies without relying on LLM parametric memory.

## Tech Stack

- **Framework:** LangChain (`langchain-huggingface`, `langchain-chroma`)
- **LLM Engine:** Groq API (`qwen/qwen3.8-27b`)
- **Vector Database:** ChromaDB
- **Embedding Model:** HuggingFace `all-MiniLM-L6-v2`
- **User Interface:** Streamlit & Terminal CLI
- **Environment:** Python 3.8+ on Ubuntu

## Project Architecture

Markdown
# ==============================================================================
# SECTION 1: TECHNICAL README.md
# Save the content below as 'README.md' in your project root folder
# ==============================================================================

# Rwanda Agricultural Statistics RAG Pipeline

A Retrieval-Augmented Generation (RAG) system for querying official Rwandan agricultural survey data and statistical reports using vector search and large language model inference.

## Overview

This application indexes official agricultural datasets (including MINAGRI livestock reports, NAEB coffee and tea statistics, and NISR survey data) into a local vector database. Users can submit natural language queries to retrieve context-specific metrics, yields, export volumes, and policy subsidies without relying on LLM parametric memory.

## Tech Stack

- **Framework:** LangChain (`langchain-huggingface`, `langchain-chroma`)
- **LLM Engine:** Groq API (`qwen/qwen3.8-27b`)
- **Vector Database:** ChromaDB
- **Embedding Model:** HuggingFace `all-MiniLM-L6-v2`
- **User Interface:** Streamlit & Terminal CLI
- **Environment:** Python 3.8+ on Ubuntu

## Project Architecture

User Query ──► HuggingFace Embeddings ──► ChromaDB Vector Search (Top-k Chunks)
                                                    │
                                                    ▼
Response ◄── Groq (Qwen 3.8 27B) ◄── Context Injection Prompt
System Requirements & Setup
Prerequisites
Python 3.8 or higher

System dependency: pysqlite3-binary (required for SQLite version compatibility on Ubuntu 20.04)

Groq API Key

Installation
Clone the repository:

Bash
git clone [https://github.com/thelmaBangsi/rwanda-agri-rag.git](https://github.com/thelmaBangsi/rwanda-agri-rag.git)
cd rwanda-agri-rag
Create and activate a virtual environment:

Bash
python3 -m venv venv
source venv/bin/activate
Install dependencies:

Bash
pip install -r requirements.txt
pip install pysqlite3-binary
Environment Variables:
Create a .env file in the root directory:

Code snippet
GROQ_API_KEY=your_groq_api_key_here
Usage
Interactive Terminal Interface
To launch the command-line chat session:

Bash
python app.py
Streamlit Web Dashboard
To start the graphical presentation interface:

Bash
streamlit run ui.py
Repository Structure
Plaintext
├── app.py              # Main application pipeline and CLI loop
├── ui.py               # Streamlit web interface
├── data/               # Source text documents and agricultural statistics
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
└── README.md           # Technical documentation
