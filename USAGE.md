# Memory Alpha RAG CLI Usage Guide

The Memory Alpha RAG CLI allows you to chat with the Star Trek Memory Alpha database using ChromaDB for retrieval and Ollama for generation.

## Prerequisites

1. ChromaDB database must be created (using `pipeline/10-extract-memoryalpha-data`)
2. Ollama must be running with a compatible model (currently using `gemma:2b`)

## Usage

### Interactive Chat Mode

Start an interactive conversation:

```bash
python3 lcars/cli.py
```

Or using the wrapper:

```bash
python3 chat.py
```

This will start an interactive session where you can ask questions about Star Trek. Type `quit`, `exit`, or `q` to exit.

Example session:
```
🖖 Welcome to the Memory Alpha RAG Chat!
Ask me anything about Star Trek. Type 'quit' or 'exit' to leave.

👤 You: Who is Captain Kirk?
🔍 Searching Memory Alpha...
📚 Found 3 relevant articles:
  • James T. Kirk
  • Kirk family
  • James Kirk (mirror)

🤖 Generating response...

🤖 Memory Alpha: James Tiberius Kirk is a captain in the Star Trek universe...
```

### Single Question Mode

Ask a single question and get an answer:

```bash
python3 lcars/cli.py --question "Who is Captain Kirk?"
```

### Command Line Options

- `--chroma-db PATH`: Path to ChromaDB database (default: `/data/enmemoryalpha_db`)
- `--ollama-url URL`: Ollama API URL (default: `http://ollama:11434`)
- `--model MODEL`: Ollama model to use (default: `gemma:2b`)
- `--question QUESTION`: Ask a single question and exit

### Examples

```bash
# Use a different model
python3 lcars/cli.py --model "llama2:7b"

# Use a different database path
python3 lcars/cli.py --chroma-db "./my_custom_db"

# Use a different Ollama instance
python3 lcars/cli.py --ollama-url "http://localhost:11434"

# Ask a specific question
python3 lcars/cli.py -q "What is the Prime Directive?"
```

## How It Works

1. **Retrieval**: The CLI searches the ChromaDB database for documents relevant to your question
2. **Context Building**: The most relevant articles are selected and their content is extracted
3. **Generation**: The context and your question are sent to Ollama to generate a comprehensive answer
4. **Response**: The AI provides an answer based on the Memory Alpha database content

## Troubleshooting

- **"model not found"**: Make sure the specified model is available in Ollama (`curl http://ollama:11434/api/tags`)
- **"database error"**: Check that ChromaDB files have correct permissions and the path is accessible
- **"connection error"**: Verify that Ollama is running and accessible at the specified URL

## Features

- ✅ Interactive chat mode
- ✅ Single question mode  
- ✅ Semantic search through 37,000+ Memory Alpha articles
- ✅ Context-aware responses using RAG (Retrieval Augmented Generation)
- ✅ Configurable models and endpoints
- ✅ Source article citations
