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
