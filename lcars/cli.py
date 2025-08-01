#!/usr/bin/env python3
"""
Memory Alpha RAG CLI - Command line interface
"""

import os
import argparse

# Handle both direct execution and module import
from memoryalpha_rag import MemoryAlphaRAG, ThinkingMode


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
    parser.add_argument("--thinking-mode", 
                       choices=["disabled", "quiet", "verbose"],
                       default="quiet",
                       help="Control reasoning display: disabled (no thinking tokens), quiet (show 'thinking...'), verbose (show full reasoning)")
    parser.add_argument("--show-performance", 
                       action="store_true",
                       help="Show performance metrics including tokens/second, thinking time, and response time")
    
    args = parser.parse_args()
    
    # Initialize the RAG system
    rag = MemoryAlphaRAG(
        chroma_db_path=args.chroma_db,
        ollama_url=args.ollama_url,
        model=args.model,
        thinking_mode=ThinkingMode(args.thinking_mode),
        show_performance=args.show_performance
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
