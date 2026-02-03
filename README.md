# RAG Chatbot for PDFs

This project provides a lightweight Retrieval-Augmented Generation (RAG) chatbot that indexes PDF documents and answers questions using the most relevant chunks. It uses TF-IDF retrieval by default and can call the OpenAI API if an API key is available.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python -m rag_chatbot.cli --pdfs path/to/file.pdf
```

Ask questions in the interactive prompt. If `OPENAI_API_KEY` is set, the chatbot will answer with the OpenAI model; otherwise it will return the most relevant context snippets.

## Configuration

- `--chunk-size`: number of words per chunk (default 800)
- `--overlap`: overlap between chunks (default 120)
- `--top-k`: number of retrieved chunks (default 4)
- `--model`: OpenAI model to use (default `gpt-4o-mini`)
