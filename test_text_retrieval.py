#!/usr/bin/env python3

import os
import sys
import random

import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

import chromadb
from chromadb.config import Settings
from pathlib import Path
from sentence_transformers import SentenceTransformer
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

# Settings
DB_PATH = "data/enmemoryalpha_db"
COLLECTION_NAME = "memoryalpha"
SAMPLE_CHAR_LENGTH = 200
TOP_K = 5

# Connect to ChromaDB
client = chromadb.PersistentClient(path=DB_PATH, settings=Settings(allow_reset=False))
collection = client.get_collection(COLLECTION_NAME)

# Initialize CLIP model
print("🤖 Loading CLIP model for query embedding...")
clip_model = SentenceTransformer('clip-ViT-B-32')
print("✅ CLIP model loaded.")

# Get all text documents
print("🔍 Fetching all text documents...")
results = collection.get(where={"content_type": "text"}, include=["documents", "metadatas", "uris"])
documents = results["documents"]
metadatas = results["metadatas"]
uris = results["uris"]

if not documents:
    print("❌ No text documents found in the collection.")
    sys.exit(1)

# Pick a random article
idx = random.randint(0, len(documents) - 1)
article_text = documents[idx]
article_meta = metadatas[idx]
article_id = uris[idx]

print(f"\n📄 Randomly selected article: {article_meta.get('title', article_id)}")

# Pick a random part of the article for search
if len(article_text) > SAMPLE_CHAR_LENGTH:
    start = random.randint(0, len(article_text) - SAMPLE_CHAR_LENGTH)
    search_text = article_text[start:start+SAMPLE_CHAR_LENGTH]
else:
    search_text = article_text

print(f"🔎 Search text sample:\n{search_text}\n")

# Perform the search
query_embedding = clip_model.encode(search_text)
search_results = collection.query(
    query_embeddings=[query_embedding],
    n_results=TOP_K,
    where={"content_type": "text"}
)

retrieved_docs = search_results["documents"][0]
retrieved_metas = search_results["metadatas"][0]
retrieved_ids = search_results["ids"][0]
retrieved_distances = search_results["distances"][0]

print("Results:")
for i, (doc, meta, doc_id, dist) in enumerate(zip(retrieved_docs, retrieved_metas, retrieved_ids, retrieved_distances)):
    print(f"\n{i+1}. Title: {meta.get('title', doc_id)}")
    print(f"   Similarity Score: {1-dist:.4f}")
    print(f"   Preview: {doc[:150]}...")
    if doc_id == article_id:
        print("   ✅ This is the original article!")

# Check if the original article was retrieved
if article_id in retrieved_ids:
    print("\n✅ Test passed: The original article was retrieved in the top results.")
else:
    print("\n❌ Test failed: The original article was NOT retrieved in the top results.")
