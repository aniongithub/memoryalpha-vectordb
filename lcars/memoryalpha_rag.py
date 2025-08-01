#!/usr/bin/env python3
"""
Memory Alpha RAG system - Core functionality
"""

import os
import sys
import requests
import json
import re
import time
from typing import List, Dict, Any

# Handle both direct execution and module import
from prompts import get_system_prompt, get_user_prompt
from thinking_mode import ThinkingMode

# Handle sqlite3 compatibility for ChromaDB
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

import chromadb
from chromadb.config import Settings

class MemoryAlphaRAG:
    def __init__(self, 
                 chroma_db_path: str = "/data/enmemoryalpha_db",
                 ollama_url: str = None,
                 model: str = None,
                 thinking_mode: ThinkingMode = ThinkingMode.QUIET,
                 show_performance: bool = False):
        """Initialize the Memory Alpha RAG system."""
        
        self.chroma_db_path = chroma_db_path
        self.ollama_url = ollama_url or os.getenv("OLLAMA_URL")
        self.model = model or os.getenv("DEFAULT_MODEL")
        self.thinking_mode = thinking_mode
        self.show_performance = show_performance
        
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
            print(f"✅ Using model '{self.model}' at {self.ollama_url}")
            
            # Display thinking mode status
            if self.thinking_mode == ThinkingMode.VERBOSE:
                print("🧠 Thinking mode: VERBOSE (showing raw reasoning)")
            elif self.thinking_mode == ThinkingMode.QUIET:
                print("🔕 Thinking mode: QUIET (hiding reasoning, showing 'thinking...')")
            else:
                print("🚫 Thinking mode: DISABLED (no reasoning tokens)")
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
        """Generate LCARS-style response using Ollama with performance tracking"""
        
        # Performance tracking variables
        start_time = time.time()
        first_token_time = None
        thinking_start_time = None
        thinking_end_time = None
        token_count = 0
        thinking_token_count = 0
        
        # Build rich context
        context_text = ""
        for i, doc in enumerate(context_docs):
            context_text += f"=== Record {i+1}: {doc['title']} ===\n"
            context_text += f"{doc['content'][:1200]}\n\n"
        
        # Get prompts from the prompts module
        system_prompt = get_system_prompt(self.thinking_mode)
        user_prompt = get_user_prompt(context_text, query)

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
                thinking_mode = False
                thinking_buffer = ""
                
                print("🤖 LCARS: ", end="", flush=True)
                
                # Process streaming response
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            if 'response' in chunk:
                                token = chunk['response']
                                full_response += token
                                token_count += len(token.split())  # Approximate token count
                                
                                # Record first token time
                                if first_token_time is None:
                                    first_token_time = time.time()
                                
                                # Handle thinking tokens based on mode
                                if self.thinking_mode == ThinkingMode.DISABLED:
                                    # No thinking mode, just print everything
                                    print(token, end="", flush=True)
                                elif self.thinking_mode == ThinkingMode.VERBOSE:
                                    # Show all tokens including thinking content
                                    print(token, end="", flush=True)
                                else:  # QUIET mode
                                    # Hide thinking content, show "thinking..." instead
                                    if thinking_mode:
                                        thinking_buffer += token
                                        thinking_token_count += len(token.split())
                                        if "</think>" in thinking_buffer:
                                            # End of thinking, show completion and start regular output
                                            thinking_end_time = time.time()
                                            print(" ⟦ analysis complete ⟧\n", end="", flush=True)
                                            thinking_mode = False
                                            # Extract any content after </think>
                                            remaining = thinking_buffer.split("</think>", 1)
                                            if len(remaining) > 1 and remaining[1]:
                                                print(remaining[1], end="", flush=True)
                                            thinking_buffer = ""
                                    elif "<think>" in token:
                                        # Start of thinking mode
                                        thinking_start_time = time.time()
                                        thinking_mode = True
                                        thinking_buffer = token
                                        # Print any content before <think>
                                        before_think = token.split("<think>", 1)[0]
                                        if before_think:
                                            print(before_think, end="", flush=True)
                                        # Immediately show "thinking..." when we start thinking
                                        print("thinking...", end="", flush=True)
                                    else:
                                        # Regular token, print it
                                        print(token, end="", flush=True)
                                
                            # Check if this is the final chunk
                            if chunk.get('done', False):
                                break
                        except json.JSONDecodeError:
                            continue
                
                end_time = time.time()
                print()  # New line after streaming is complete
                
                # Show performance metrics if enabled
                if self.show_performance:
                    total_time = end_time - start_time
                    time_to_first_token = first_token_time - start_time if first_token_time else 0
                    tokens_per_second = token_count / total_time if total_time > 0 else 0
                    
                    print(f"\n📊 Performance Metrics:")
                    print(f"   ⏱️  Total time: {total_time:.2f}s")
                    print(f"   🚀 Time to first token: {time_to_first_token:.2f}s")
                    print(f"   🔢 Total tokens: {token_count}")
                    print(f"   ⚡ Tokens/second: {tokens_per_second:.1f}")
                    
                    if thinking_start_time and thinking_end_time:
                        thinking_time = thinking_end_time - thinking_start_time
                        print(f"   🧠 Thinking time: {thinking_time:.2f}s")
                        print(f"   🤔 Thinking tokens: {thinking_token_count}")
                        if thinking_time > 0:
                            thinking_tps = thinking_token_count / thinking_time
                            print(f"   💭 Thinking tokens/second: {thinking_tps:.1f}")
                
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
