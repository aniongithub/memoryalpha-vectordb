# Memory Alpha Vector DB Pipeline 🖖

[![CI Pipeline](https://github.com/aniongithub/memoryalpha_chromadb/actions/workflows/ci-pipeline.yml/badge.svg)](https://github.com/aniongithub/memoryalpha_chromadb/actions/workflows/ci-pipeline.yml)

This repository provides a reproducible pipeline for downloading, parsing, and publishing up-to-date vector database (ChromaDB) dumps of the complete Star Trek Memory Alpha wiki. These vector DB artifacts are intended for use in downstream projects, such as search, RAG, or LLM applications.

## Features

- **Automated Data Pipeline**: Download, extract, and process the latest Memory Alpha XML dump
- **Canon-only ingestion (Tier 2 strict)**: Embeds only in-universe canon articles; all real-world / non-canon pages (episodes, films, novels, comics, video games, reference books, actors, staff, non-canon character pages) are skipped
- **ChromaDB Vector Database**: Converts all articles into a persistent ChromaDB vector DB
- **Compressed Artifact**: Publishes a compressed, ready-to-use DB for easy distribution
- **CI/CD Workflows**: GitHub Actions for validation and release artifact publishing
- **Containerized**: All steps run in Docker or Dev Container for reproducibility


## Quick Start

The easiest way to use the Memory Alpha vector database is to download the latest release artifact:

1. Go to the [Releases page](https://github.com/aniongithub/memoryalpha_chromadb/releases)
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
git clone https://github.com/aniongithub/memoryalpha_chromadb.git
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
│   ├── 10-extract-memoryalpha-data   # Parse, canon-filter, and create ChromaDB
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

## Canon Filtering

By default the pipeline performs **Tier 2 strict canon filtering**: only
in-universe (canon) articles are embedded into the vector database.

Memory Alpha tags every real-world / non-canon article with the `{{real world}}`
template (also spelled `{{realworld}}`) at the top of its wikitext, and non-canon
character pages with `{{Non canon character page}}`. In-universe canon articles
do not carry these markers. During extraction (`pipeline/10-extract-memoryalpha-data`),
any page whose wikitext contains one of these templates is skipped **before** any
image download or embedding work — so episodes, films, novels, comics, video
games, reference books, actor/staff pages, and non-canon character pages are all
excluded. This keeps retrieval focused on in-universe content (e.g. "Who is
Captain Picard?" returns his in-universe biography rather than novels or
reference books).

To disable the filter and ingest every article (legacy behavior), set the
`CANON_ONLY` environment variable to `0`:

```bash
CANON_ONLY=0 ./data-pipeline-docker.sh
```

## Images

Each embedded page also contributes up to a few images to a separate CLIP image
collection. Memory Alpha's `Special:FilePath` endpoint sits behind Cloudflare,
which serves an HTTP 403 challenge to datacenter IPs (including GitHub Actions
runners), so it cannot be used to fetch images in CI. Instead the pipeline asks
the MediaWiki API (`api.php`, which is not challenged) for each file's canonical
image URL and downloads it directly from the image CDN
(`static.wikia.nocookie.net`). Both the API and the CDN are reachable from
datacenter IPs, so image embedding works in CI as well as locally.

The API endpoint and request User-Agent can be overridden via the
`IMAGE_API_ENDPOINT` and `IMAGE_USER_AGENT` environment variables. If an image
can't be resolved or downloaded it is skipped; an empty image collection is a
non-fatal warning and never blocks publishing a valid text database.

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
