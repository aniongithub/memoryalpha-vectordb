# Memory Alpha Vector DB Pipeline 🖖

[![CI Pipeline](https://github.com/aniongithub/memoryalpha_chromadb/actions/workflows/ci-pipeline.yml/badge.svg?branch=main)](https://github.com/aniongithub/memoryalpha_rag/actions/workflows/ci-pipeline.yml)

This repository provides a reproducible pipeline for downloading, parsing, and publishing up-to-date vector database (ChromaDB) dumps of the complete Star Trek Memory Alpha wiki. These vector DB artifacts are intended for use in downstream projects, such as search, RAG, or LLM applications.

## Features

- **Automated Data Pipeline**: Download, extract, and process the latest Memory Alpha XML dump
- **ChromaDB Vector Database**: Converts all articles into a persistent ChromaDB vector DB
- **Compressed Artifact**: Publishes a compressed, ready-to-use DB for easy distribution
- **CI/CD Workflows**: GitHub Actions for validation and release artifact publishing
- **Containerized**: All steps run in Docker or Dev Container for reproducibility


## Quick Start

The easiest way to use the Memory Alpha vector database is to download the latest release artifact:

1. Go to the [Releases page](https://github.com/aniongithub/memoryalpha_rag/releases)
2. Download `enmemoryalpha_db.tar.gz`
3. Extract it:

   ```bash
   tar xzf enmemoryalpha_db.tar.gz
   # or
   7z x enmemoryalpha_db.tar.gz
   ```

4. Use the extracted `enmemoryalpha_db/` directory in your own ChromaDB-powered project.

### Example: Cosine Similarity Search with ChromaDB

Here's a minimal example of how to load the DB and perform a cosine similarity search:

```python
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(path="enmemoryalpha_db", settings=Settings(allow_reset=True))
collection = client.get_or_create_collection("memoryalpha")

# Example query
query = "Who is Captain Picard?"
results = collection.query(query_texts=[query], n_results=3)
for i, doc in enumerate(results["documents"][0]):
    print(f"Result {i+1}:\nTitle: {results['metadatas'][0][i]['title']}\nContent: {doc[:300]}\n---")
```

---

## Development

### Prerequisites

- [Docker](https://www.docker.com/get-started)
- [VS Code](https://code.visualstudio.com/) with [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) (optional)

### 1. Clone and Open

```bash
git clone https://github.com/aniongithub/memoryalpha_rag.git
cd memoryalpha_rag
```

Open in VS Code and reopen in container if desired.

### 2. Run the Pipeline

```bash
# This will download, extract, vectorize, and compress the Memory Alpha database
./data-pipeline-docker.sh
```

The result will be a compressed ChromaDB artifact at:

```
data/enmemoryalpha_db.tar.gz
```

You can now use `data/enmemoryalpha_db.tar.gz` in your own projects. Decompress and mount as needed for downstream applications.

## Project Structure

```
memoryalpha_rag/
├── pipeline/                  # Data processing pipeline scripts
│   ├── 00-download-memory-alpha      # Download Memory Alpha dump
│   ├── 10-extract-memoryalpha-data   # Parse and create ChromaDB
│   ├── 20-compress-memoryalpha-db    # Compress database
│   └── pipeline.Dockerfile           # Pipeline container
├── data/                      # Data directory (gitignored)
│   ├── enmemoryalpha_pages_current.xml    # Raw Memory Alpha dump
│   ├── enmemoryalpha_db/                  # ChromaDB database
│   └── enmemoryalpha_db.tar.gz            # Compressed database
├── data-pipeline-docker.sh    # Pipeline execution script
├── .github/workflows/         # CI/CD workflows
└── README.md                  # This file
```

## CI/CD

- **Pull Request to main**: Runs the pipeline as a CI check (no artifact published)
- **Release Published**: Runs the pipeline and uploads the compressed DB as a release asset

See `.github/workflows/` for details.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- **Memory Alpha** - The Star Trek wiki providing the comprehensive database
- **Wikia/Fandom** - Hosting the Memory Alpha XML dumps
- **ChromaDB** - Vector database for semantic search

---

**Live long and prosper!** 🖖
