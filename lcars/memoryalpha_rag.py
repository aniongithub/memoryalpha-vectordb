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
import warnings
import logging
import numpy as np
from typing import List, Dict, Any

# Suppress the specific transformers deprecation warning
warnings.filterwarnings("ignore", message=".*encoder_attention_mask.*is deprecated.*", category=FutureWarning)

logger = logging.getLogger(__name__)

# Enhanced input handling
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter

# Cross-encoder for re-ranking
from sentence_transformers import CrossEncoder, SentenceTransformer

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
                 show_performance: bool = False,
                 enable_streaming: bool = True,
                 max_history_turns: int = 5,
                 rerank_method: str = "cross-encoder"):
        """Initialize the Memory Alpha RAG system."""
        
        self.chroma_db_path = chroma_db_path
        self.ollama_url = ollama_url or os.getenv("OLLAMA_URL")
        self.model = model or os.getenv("DEFAULT_MODEL")
        self.thinking_mode = thinking_mode
        self.show_performance = show_performance
        self.enable_streaming = enable_streaming
        self.rerank_method = rerank_method
        
        # Conversation history for context
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history_turns = max_history_turns  # Keep last N exchanges
        
        if not self.ollama_url:
            logger.error("OLLAMA_URL environment variable is required")
            sys.exit(1)
        if not self.model:
            logger.error("DEFAULT_MODEL environment variable is required")
            sys.exit(1)
        
        # Initialize re-ranking models based on selected method
        if self.rerank_method == "cross-encoder":
            logger.info("Loading cross-encoder re-ranker...")
            try:
                # Try the newer BGE reranker first, fallback to the base model
                try:
                    self.cross_encoder = CrossEncoder('BAAI/bge-reranker-v2-m3')
                    logger.info("Cross-encoder re-ranker loaded (BGE v2-m3)")
                except Exception:
                    self.cross_encoder = CrossEncoder('BAAI/bge-reranker-base')
                    logger.info("Cross-encoder re-ranker loaded (BGE base)")
                self.embedding_model = None
            except Exception as e:
                logger.warning(f"Could not load cross-encoder ({e}), falling back to cosine similarity ranking")
                self.rerank_method = "cosine"
                self.cross_encoder = None
        
        if self.rerank_method == "cosine":
            logger.info("Loading embedding model for cosine similarity re-ranking...")
            try:
                # Use a lightweight, fast embedding model for cosine similarity
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Embedding model loaded (all-MiniLM-L6-v2) for cosine similarity")
                self.cross_encoder = None
            except Exception as e:
                logger.error(f"Could not load embedding model ({e}), falling back to distance-only ranking")
                self.embedding_model = None
                self.cross_encoder = None
        
        # Initialize ChromaDB client
        try:
            self.client = chromadb.PersistentClient(
                path=chroma_db_path, 
                settings=Settings(allow_reset=False)
            )
            self.collection = self.client.get_collection("memoryalpha")
            logger.info(f"Connected to Memory Alpha at {chroma_db_path}")
            logger.info(f"Memory Core contains {self.collection.count()} documents")
            logger.info(f"Using model '{self.model}' at {self.ollama_url}")
            
            # Display thinking mode status
            if self.thinking_mode == ThinkingMode.VERBOSE:
                logger.info("Thinking mode: VERBOSE (showing raw reasoning)")
            elif self.thinking_mode == ThinkingMode.QUIET:
                logger.info("Thinking mode: QUIET (hiding reasoning, showing 'thinking...')")
            else:
                logger.info("Thinking mode: DISABLED (no reasoning tokens)")
            
            # Display re-ranking method status
            if self.cross_encoder:
                logger.info("Re-ranking method: Cross-encoder (slower but more accurate)")
            elif self.embedding_model:
                logger.info("Re-ranking method: Cosine similarity (faster)")
            else:
                logger.info("Re-ranking method: Distance-only (fallback)")
            
            # Display conversation history status
            if self.max_history_turns > 0:
                logger.info(f"Conversation history: Enabled (keeping last {self.max_history_turns} exchanges)")
            else:
                logger.info("Conversation history: Disabled")
                
            # Display streaming mode status
            if self.enable_streaming:
                logger.info("Streaming mode is enabled")
            else:
                logger.info("Streaming mode is disabled")
            
            # Warm up the model to keep it loaded in memory
            self._warm_up_model()
                
        except Exception as e:
            logger.error(f"Error connecting to Memory Alpha: {e}")
            sys.exit(1)
    
    def _cosine_similarity(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and document embeddings"""
        # Normalize embeddings
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        
        # Compute cosine similarity
        similarities = np.dot(doc_norms, query_norm)
        return similarities
    
    def _warm_up_model(self):
        """Warm up the model and keep it loaded in memory indefinitely"""
        try:
            logger.info(f"Warming up model '{self.model}' and keeping it loaded...")
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": "System ready.",
                    "keep_alive": -1,  # Keep model loaded indefinitely
                    "stream": False
                },
                timeout=30
            )
            if response.status_code == 200:
                logger.info("Model warmed up and will stay loaded in memory")
            else:
                logger.warning(f"Model warmup response: {response.status_code}")
        except Exception as e:
            logger.warning(f"Could not warm up model: {e}")
    
    
    def smart_search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using ChromaDB
        """
        search_start = time.time()
        
        # Use the query as-is without artificial expansions
        search_queries = [query]
        
        # Collect results from searches
        all_documents = []
        seen_ids = set()
        candidates_per_query = 8 if self.show_performance else 10
        
        db_search_start = time.time()
        for search_query in search_queries:
            try:
                results = self.collection.query(
                    query_texts=[search_query],
                    n_results=candidates_per_query
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
                logger.error(f"Search error for '{search_query}': {e}")
                continue
        
        db_search_time = time.time() - db_search_start
        
        # Sort by relevance (lower distance = more relevant)
        all_documents.sort(key=lambda x: x['distance'])
        
        # Skip re-ranking only if no re-ranking models are available
        if not self.cross_encoder and not self.embedding_model:
            candidates = all_documents[:n_results]
            rerank_time = 0
        else:
            # Limit to top candidates for re-ranking
            candidates = all_documents[:15]  # Reduced from 25 to 15
            
            # Re-rank using selected method
            rerank_time = 0
            if len(candidates) > 1:
                try:
                    rerank_start = time.time()
                    
                    if self.rerank_method == "cross-encoder" and self.cross_encoder:
                        # Cross-encoder re-ranking (original method)
                        pairs = []
                        for doc in candidates:
                            # Use first 300 chars of content for efficiency (reduced from 500)
                            content_snippet = doc['content'][:300]
                            pairs.append([query, content_snippet])
                        
                        # Get cross-encoder scores
                        scores = self.cross_encoder.predict(pairs)
                        
                        # Add scores to documents and re-sort
                        for i, doc in enumerate(candidates):
                            doc['cross_encoder_score'] = float(scores[i])
                        
                        # Sort by cross-encoder score (higher = more relevant)
                        candidates.sort(key=lambda x: x['cross_encoder_score'], reverse=True)
                        
                    elif self.rerank_method == "cosine" and self.embedding_model:
                        # Cosine similarity re-ranking (new faster method)
                        # Get query embedding
                        query_embedding = self.embedding_model.encode([query])
                        
                        # Get document embeddings
                        doc_texts = []
                        for doc in candidates:
                            # Use first 300 chars of content for efficiency
                            content_snippet = doc['content'][:300]
                            doc_texts.append(content_snippet)
                        
                        doc_embeddings = self.embedding_model.encode(doc_texts)
                        
                        # Compute cosine similarities
                        similarities = self._cosine_similarity(query_embedding[0], doc_embeddings)
                        
                        # Add scores to documents and re-sort
                        for i, doc in enumerate(candidates):
                            doc['cosine_score'] = float(similarities[i])
                        
                        # Sort by cosine similarity (higher = more relevant)
                        candidates.sort(key=lambda x: x['cosine_score'], reverse=True)
                    
                    rerank_time = time.time() - rerank_start
                    
                except Exception as e:
                    logger.warning(f"Re-ranking failed: {e}, using distance-only ranking")
            
            candidates = candidates[:n_results]
        
        total_search_time = time.time() - search_start
        
        if self.show_performance:
            print(f"  🔍 DB queries: {db_search_time:.2f}s")
            if rerank_time > 0:
                method_name = "cross-encoder" if self.rerank_method == "cross-encoder" else "cosine similarity"
                print(f"  🎯 Re-ranking ({method_name}): {rerank_time:.2f}s")
            print(f"  📊 Total search: {total_search_time:.2f}s")
        
        return candidates[:n_results]
    
    def add_to_conversation_history(self, question: str, answer: str):
        """Add a question-answer pair to conversation history"""
        # Clean the answer by removing ANSI codes and extra formatting
        clean_answer = re.sub(r'\033\[[0-9;]*m', '', answer)  # Remove ANSI color codes
        clean_answer = clean_answer.replace("LCARS: ", "").strip()
        
        # Remove thinking blocks from the answer for history
        while "<think>" in clean_answer and "</think>" in clean_answer:
            start = clean_answer.find("<think>")
            end = clean_answer.find("</think>") + len("</think>")
            clean_answer = clean_answer[:start] + clean_answer[end:]
        
        # Remove "Working..." prefix if present
        if clean_answer.startswith("Working..."):
            clean_answer = clean_answer[10:].strip()
        
        self.conversation_history.append({
            "question": question,
            "answer": clean_answer
        })
        
        # Keep only the last max_history_turns exchanges
        if len(self.conversation_history) > self.max_history_turns:
            self.conversation_history = self.conversation_history[-self.max_history_turns:]
    
    def clear_conversation_history(self):
        """Clear the conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def show_conversation_history(self):
        """Display the current conversation history"""
        if not self.conversation_history:
            logger.info("No conversation history")
            return
        
        logger.info(f"Conversation History ({len(self.conversation_history)} exchanges):")
        for i, exchange in enumerate(self.conversation_history, 1):
            logger.info(f"  Q{i}: {exchange['question'][:50]}{'...' if len(exchange['question']) > 50 else ''}")
            logger.info(f"  A{i}: {exchange['answer'][:50]}{'...' if len(exchange['answer']) > 50 else ''}")
    
    def generate_response(self, query: str, context_docs: List[Dict[str, Any]], max_tokens: int = 2048) -> str:
        """Generate LCARS-style response using Ollama with performance tracking
        
        Args:
            query: The user's question
            context_docs: List of relevant documents for context
            max_tokens: Maximum number of tokens to generate (default: 2048)
        """
        
        # Performance tracking variables
        start_time = time.time()
        first_token_time = None
        thinking_start_time = None
        thinking_end_time = None
        token_count = 0
        thinking_token_count = 0
        
        # Build rich context
        context_build_start = time.time()
        context_text = ""
        # Further reduce context size for performance mode
        char_limit = 600 if self.show_performance else 800
        for i, doc in enumerate(context_docs):
            context_text += f"=== Record {i+1}: {doc['title']} ===\n"
            context_text += f"{doc['content'][:char_limit]}\n\n"
        
        # Get prompts from the prompts module
        system_prompt = get_system_prompt(self.thinking_mode)
        # Use shorter context when performance tracking is enabled
        conversation_context = self.conversation_history if not self.show_performance else []
        user_prompt = get_user_prompt(context_text, query, conversation_context)
        context_build_time = time.time() - context_build_start

        if self.show_performance:
            print(f"  📝 Context build: {context_build_time:.2f}s")
            print(f"  📏 Prompt length: {len(system_prompt + user_prompt):,} chars")

        try:
            request_start = time.time()
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": f"{system_prompt}\n\n{user_prompt}",
                    "stream": self.enable_streaming,  # Use the streaming flag
                    "options": {
                        "temperature": 0.3,  # Lower temperature for more consistent responses
                        "top_p": 0.8,
                        "num_predict": max_tokens  # Use the parameter passed to the method
                    }
                },
                timeout=120,
                stream=self.enable_streaming  # Enable streaming in requests based on flag
            )
            request_time = time.time() - request_start
            
            if self.show_performance:
                print(f"  🌐 Request setup: {request_time:.2f}s")
            
            if response.status_code == 200:
                full_response = ""
                thinking_mode = False
                thinking_buffer = ""
                
                print("LCARS: ", end="", flush=True)
                
                if self.enable_streaming:
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
                                        # Show all tokens including thinking content in gray for thinking sections
                                        if "<think>" in token:
                                            # Start of thinking - print content before <think> normally, then switch to gray
                                            before_think = token.split("<think>", 1)[0]
                                            if before_think:
                                                print(before_think, end="", flush=True)
                                            print("\033[90m<think>", end="", flush=True)  # Start gray text
                                            after_think = token.split("<think>", 1)[1] if "<think>" in token else ""
                                            if after_think:
                                                print(after_think, end="", flush=True)
                                        elif "</think>" in token:
                                            # End of thinking - print thinking content in gray, then switch back to normal
                                            before_end = token.split("</think>", 1)[0]
                                            if before_end:
                                                print(before_end, end="", flush=True)
                                            print("</think>\033[0m", end="", flush=True)  # End gray text and reset
                                            after_end = token.split("</think>", 1)[1] if "</think>" in token else ""
                                            if after_end:
                                                print(after_end, end="", flush=True)
                                        else:
                                            # Regular token in verbose mode
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
                                            # Continue to next token while in thinking mode
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
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to decode JSON chunk: {e}")
                                continue
                            except Exception as e:
                                logger.error(f"Error processing streaming chunk: {e}")
                                continue
                else:
                    # Non-streaming response
                    response_data = response.json()
                    if 'response' in response_data:
                        full_response = response_data['response']
                        token_count = len(full_response.split())  # Approximate token count
                        first_token_time = time.time()  # Set first token time immediately
                        
                        # Handle thinking content for non-streaming
                        if self.thinking_mode == ThinkingMode.DISABLED:
                            print(full_response, end="", flush=True)
                        elif self.thinking_mode == ThinkingMode.VERBOSE:
                            # Show all content including thinking in gray
                            output = full_response
                            output = output.replace("<think>", "\033[90m<think>")
                            output = output.replace("</think>", "</think>\033[0m")
                            print(output, end="", flush=True)
                        else:  # QUIET mode
                            # Remove thinking content and show processed response
                            if "<think>" in full_response and "</think>" in full_response:
                                thinking_start_time = time.time()
                                # Extract thinking content for token counting
                                thinking_parts = full_response.split("<think>")[1].split("</think>")[0]
                                thinking_token_count = len(thinking_parts.split())
                                thinking_end_time = time.time()
                                
                                # Remove thinking tags and content
                                processed_response = full_response
                                while "<think>" in processed_response and "</think>" in processed_response:
                                    start = processed_response.find("<think>")
                                    end = processed_response.find("</think>") + len("</think>")
                                    processed_response = processed_response[:start] + processed_response[end:]
                                
                                print("thinking... ⟦ analysis complete ⟧\n", end="", flush=True)
                                print(processed_response.strip(), end="", flush=True)
                            else:
                                print(full_response, end="", flush=True)
                
                end_time = time.time()
                print()  # New line after streaming is complete
                
                # Show performance metrics if enabled
                if self.show_performance:
                    total_time = end_time - start_time
                    time_to_first_token = first_token_time - start_time if first_token_time else 0
                    tokens_per_second = token_count / total_time if total_time > 0 else 0
                    
                    print(f"\nPerformance Metrics:")
                    print(f"   Total time: {total_time:.2f}s")
                    print(f"   Time to first token: {time_to_first_token:.2f}s")
                    print(f"   Total tokens: {token_count}")
                    print(f"   Tokens/second: {tokens_per_second:.1f}")
                    
                    if thinking_start_time and thinking_end_time:
                        thinking_time = thinking_end_time - thinking_start_time
                        print(f"   Thinking time: {thinking_time:.2f}s")
                        print(f"   Thinking tokens: {thinking_token_count}")
                        if thinking_time > 0:
                            thinking_tps = thinking_token_count / thinking_time
                            print(f"   Thinking tokens/second: {thinking_tps:.1f}")
                
                return full_response
            else:
                return f"❌ Error from Ollama API: {response.status_code} - {response.text}"
                
        except requests.exceptions.RequestException as e:
            return f"❌ Error connecting to Ollama: {e}"
    
    def answer_question(self, question: str, max_tokens: int = 2048) -> str:
        """Answer a single question with improved search and response
        
        Args:
            question: The user's question
            max_tokens: Maximum number of tokens to generate (default: 2048)
        """
        
        # Start timing the entire process
        total_start = time.time()
        
        # Search for relevant documents - reduce count when performance is enabled for faster responses
        search_results = 6 if self.show_performance else 10
        search_start = time.time()
        context_docs = self.smart_search(question, n_results=search_results)
        search_time = time.time() - search_start
        
        if not context_docs:
            return "Unable to comply. No relevant information found in Memory Alpha records."
        
        # Show what was found
        print(f"📚 Located {len(context_docs)} relevant records:")
        for doc in context_docs:
            print(f"  • {doc['title']}")
        
        if self.show_performance:
            print(f"🔍 Search completed in {search_time:.2f}s")
        
        print("\nProcessing query...")
        
        # Time the response generation
        generation_start = time.time()
        response = self.generate_response(question, context_docs, max_tokens)
        generation_time = time.time() - generation_start
        
        total_time = time.time() - total_start
        
        if self.show_performance:
            print(f"⚡ Generation took {generation_time:.2f}s")
            print(f"📊 Total query time: {total_time:.2f}s")
        
        # Add to conversation history
        self.add_to_conversation_history(question, response)
        
        return response
    
    def chat_loop(self, default_max_tokens: int = 2048):
        """Interactive chat loop with enhanced input handling
        
        Args:
            default_max_tokens: Default maximum tokens for responses (default: 2048)
        """
        print("\n" + "="*60)
        print("🖖 LIBRARY COMPUTER ACCESS AND RETRIEVAL SYSTEM")
        print("   Memory Alpha Database Interface")
        print("="*60)
        
        print("LCARS Ready. Enhanced input mode enabled.")
        print("Use Ctrl+C to exit, Arrow keys to navigate, Tab for completion.")
        print("Special commands: 'clear history', 'show history'")
        # Set up enhanced input with history and auto-suggestions
        history = InMemoryHistory()
        star_trek_commands = WordCompleter([
            'Who is', 'What is', 'Tell me about', 'Explain', 'Describe',
            'Captain Kirk', 'Captain Picard', 'Data', 'Spock', 'Enterprise',
            'USS Enterprise', 'phaser', 'transporter', 'warp drive', 'Starfleet',
            'Federation', 'Klingon', 'Vulcan', 'Romulan', 'Borg',
            'clear history', 'show history'
        ])
        
        print()
        
        while True:
            try:
                query = prompt(
                    "👤 QUERY: ",
                    history=history,
                    auto_suggest=AutoSuggestFromHistory(),
                    completer=star_trek_commands,
                    complete_style='column',
                    multiline=False
                ).strip()
                
                if query.lower() in ['quit', 'exit', 'q', 'end program', 'computer off']:
                    print("\n🖖 LCARS offline. Live long and prosper!")
                    break
                
                if query.lower() in ['clear history', 'clear']:
                    self.clear_conversation_history()
                    continue
                
                if query.lower() in ['show history', 'history']:
                    self.show_conversation_history()
                    continue
                
                if not query:
                    continue
                
                print("\nAccessing Memory Alpha databases...")
                
                # Generate response using instance streaming setting
                response = self.answer_question(query, max_tokens=default_max_tokens)
                print(f"\n")  # Extra spacing after response
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\n\n🖖 LCARS offline. Live long and prosper!")
                break
            except Exception as e:
                print(f"❌ System error: {e}")
