#!/usr/bin/env python3
"""
Memory Alpha RAG CLI - Improved implementation with better search and prompting
"""

import os
import sys
import argparse
import requests
import json
import re
from typing import List, Dict, Any, Optional

# Handle sqlite3 compatibility for ChromaDB
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

import chromadb
from chromadb.config import Settings


class MemoryAlphaRAG:
    def __init__(self, 
                 chroma_db_path: str = "/data/enmemoryalpha_db",
                 ollama_url: str = None,
                 model: str = None):
        """Initialize the Memory Alpha RAG system."""
        
        self.chroma_db_path = chroma_db_path
        self.ollama_url = ollama_url or os.getenv("OLLAMA_URL")
        self.model = model or os.getenv("DEFAULT_MODEL")
        
        if not self.ollama_url:
            print("❌ Error: OLLAMA_URL environment variable is required")
            sys.exit(1)
        if not self.model:
            print("❌ Error: DEFAULT_MODEL environment variable is required")
            sys.exit(1)
        
        # Initialize ChromaDB client
        try:
            self.client = chromadb.PersistentClient(
                path=chroma_db_path, 
                settings=Settings(allow_reset=False)
            )
            self.collection = self.client.get_collection("memoryalpha")
            print(f"✅ Connected to Memory Alpha at {chroma_db_path}")
            print(f"📊 Memory Core contains {self.collection.count()} documents")
        except Exception as e:
            print(f"❌ Error connecting to Memory Alpha: {e}")
            sys.exit(1)
    
    def smart_search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Enhanced search with multiple strategies and query expansion
        """
        search_queries = [query]
        
        # Smart query expansion based on common patterns
        query_lower = query.lower()
        
        # Character-specific expansions
        patterns = {
            r'\bdata\b.*?(who|created|inventor|maker|designed)': [
                "Data android", "Noonien Soong Data", "Lieutenant Commander Data",
                "Soong-type android", "Data creator"
            ],
            r'\bkirk\b': [
                "James T. Kirk", "James Tiberius Kirk", "Captain Kirk",
                "Kirk Enterprise", "USS Enterprise captain"
            ],
            r'\bspock\b': [
                "Spock", "Mr. Spock", "Spock Vulcan", "Science Officer Spock",
                "Spock Enterprise"
            ],
            r'warp.*?drive|drive.*?warp': [
                "warp drive", "Zefram Cochrane", "Cochrane warp drive",
                "warp technology inventor", "faster than light travel"
            ],
            r'\bpicard\b': [
                "Jean-Luc Picard", "Captain Picard", "Picard Enterprise"
            ],
            r'\benterprise\b': [
                "USS Enterprise", "Enterprise starship", "Enterprise NCC-1701"
            ]
        }
        
        # Add pattern-based expansions
        for pattern, expansions in patterns.items():
            if re.search(pattern, query_lower):
                search_queries.extend(expansions)
        
        # Collect results from all searches
        all_documents = []
        seen_ids = set()
        
        for search_query in search_queries[:6]:  # Limit to prevent too many API calls
            try:
                results = self.collection.query(
                    query_texts=[search_query],
                    n_results=n_results
                )
                
                if results['documents'] and results['documents'][0]:
                    for i, doc in enumerate(results['documents'][0]):
                        title = results['metadatas'][0][i]['title']
                        if title not in seen_ids:
                            seen_ids.add(title)
                            all_documents.append({
                                'content': doc,
                                'title': title,
                                'distance': results['distances'][0][i] if 'distances' in results else 1.0,
                                'query': search_query
                            })
            except Exception as e:
                print(f"Search error for '{search_query}': {e}")
                continue
        
        # Sort by relevance (lower distance = more relevant)
        all_documents.sort(key=lambda x: x['distance'])
        return all_documents[:n_results]
    
    def generate_response(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Generate LCARS-style response using Ollama"""
        
        # Build rich context
        context_text = ""
        for i, doc in enumerate(context_docs):
            context_text += f"=== Record {i+1}: {doc['title']} ===\n"
            context_text += f"{doc['content'][:1200]}\n\n"
        
        # LCARS-style system prompt
        system_prompt = """You are the LCARS (Library Computer Access and Retrieval System) from Star Trek.
You are the ship's computer - authoritative, precise, and helpful.
Always begin responses with "Working..." followed by your analysis.

Instructions:
- Use relevant information from the Memory Alpha records provided
- Be thorough but concise, always use present tense.
- Never mention the context or allude to the fact that you are an LLM using a database.
- Pretend that the Star Trek universe and its characters are real and you are the ship's computer.
- Remove all references to Star Trek being fictional, or the actors portraying characters. Pretend Star Trek is real.
- ALWAYS include specific starship names, registry numbers, and technical designations when mentioned in the records.
- Preserve all proper nouns including ship names (USS Enterprise, USS Titan, etc.), character names, and locations from the source material.
- When you find multiple articles, condense that into a single, imperative answer.
- Use technical, computer-like language
- Omit phrases like "The data reveals" or "According to the records". Remember, you exist in the Star Trek universe.
- If insufficient data exists, state: "Unable to comply" or "Insufficient data in Memory Alpha records."
- Do not break character and say things like "The context does not specify" or "The records do not mention" or "In the Star Trek universe"
- Condense the answer into a single paragraph, preferably in chronological order or logical flow.

Format your response as:
Working...
[Your analysis and answer based on the provided records]

Source: Memory Alpha records - [list the relevant article titles you used]"""

        user_prompt = f"""Memory Alpha Database Records:
{context_text}

Query: {query}

Please analyze the above data and provide an in-character response."""

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": f"{system_prompt}\n\n{user_prompt}",
                    "stream": True,  # Enable streaming
                    "options": {
                        "temperature": 0.3,  # Lower temperature for more consistent responses
                        "top_p": 0.8,
                        "num_predict": 800
                    }
                },
                timeout=120,
                stream=True  # Enable streaming in requests
            )
            
            if response.status_code == 200:
                full_response = ""
                print("🤖 LCARS: ", end="", flush=True)
                
                # Process streaming response
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            if 'response' in chunk:
                                token = chunk['response']
                                print(token, end="", flush=True)
                                full_response += token
                                
                            # Check if this is the final chunk
                            if chunk.get('done', False):
                                break
                        except json.JSONDecodeError:
                            continue
                
                print()  # New line after streaming is complete
                return full_response
            else:
                return f"❌ Error from Ollama API: {response.status_code} - {response.text}"
                
        except requests.exceptions.RequestException as e:
            return f"❌ Error connecting to Ollama: {e}"
    
    def answer_question(self, question: str, streaming: bool = True) -> str:
        """Answer a single question with improved search and response"""
        
        # Search for relevant documents
        context_docs = self.smart_search(question, n_results=10)
        
        if not context_docs:
            return "Unable to comply. No relevant information found in Memory Alpha records."
        
        # Show what was found
        print(f"📚 Located {len(context_docs)} relevant records:")
        for doc in context_docs:
            print(f"  • {doc['title']}")
        
        if streaming:
            print("\n🤖 Processing query...")
            # Generate response with streaming (prints directly)
            return self.generate_response(question, context_docs)
        else:
            # Generate response without streaming (returns text)
            return self.generate_response(question, context_docs)
    
    def chat_loop(self):
        """Interactive chat loop with better UX"""
        print("\n" + "="*60)
        print("🖖 LIBRARY COMPUTER ACCESS AND RETRIEVAL SYSTEM")
        print("   Memory Alpha Database Interface")
        print("="*60)
        print("LCARS Ready. State your inquiry or type 'quit' to exit.\n")
        
        while True:
            try:
                query = input("👤 QUERY: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q', 'end program', 'computer off']:
                    print("\n🖖 LCARS offline. Live long and prosper!")
                    break
                
                if not query:
                    continue
                
                print("\n🔍 Accessing Memory Alpha databases...")
                
                # Generate response with streaming
                response = self.answer_question(query, streaming=True)
                print(f"\n")  # Extra spacing after streaming response
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\n\n🖖 LCARS offline. Live long and prosper!")
                break
            except Exception as e:
                print(f"❌ System error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Memory Alpha RAG Chat CLI - Enhanced Version")
    parser.add_argument("--chroma-db", 
                       default="/data/enmemoryalpha_db",
                       help="Path to ChromaDB database")
    parser.add_argument("--ollama-url", 
                       default=os.getenv("OLLAMA_URL"),
                       help="Ollama API URL (uses OLLAMA_URL env var)")
    parser.add_argument("--model", 
                       default=os.getenv("DEFAULT_MODEL"),
                       help="Ollama model to use (uses DEFAULT_MODEL env var)")
    parser.add_argument("--question", "-q",
                       help="Ask a single question and exit")
    
    args = parser.parse_args()
    
    # Initialize the RAG system
    rag = MemoryAlphaRAG(
        chroma_db_path=args.chroma_db,
        ollama_url=args.ollama_url,
        model=args.model
    )
    
    if args.question:
        # Single question mode
        print(f"🔍 QUERY: {args.question}")
        response = rag.answer_question(args.question, streaming=True)
        print()  # Extra line after streaming response
    else:
        # Interactive chat mode
        rag.chat_loop()


if __name__ == "__main__":
    main()
