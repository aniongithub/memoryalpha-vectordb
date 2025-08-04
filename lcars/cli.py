#!/usr/bin/env python3
"""
Memory Alpha RAG CLI - Command line interface
"""

import os
import argparse
import logging
import sys

# Handle both direct execution and module import
from memoryalpha_rag import MemoryAlphaRAG, ThinkingMode


def setup_colored_logging():
    """Setup colored logging for the CLI"""
    
    class ColoredFormatter(logging.Formatter):
        """Custom formatter with colors for different log levels"""
        
        # ANSI color codes
        COLORS = {
            'DEBUG': '\033[36m',    # Cyan
            'INFO': '\033[32m',     # Green
            'WARNING': '\033[33m',  # Yellow
            'ERROR': '\033[31m',    # Red
            'CRITICAL': '\033[35m', # Magenta
        }
        RESET = '\033[0m'
        
        def format(self, record):
            # Get the original formatted message
            message = super().format(record)
            
            # Add color if we're writing to a terminal
            if sys.stderr.isatty():
                color = self.COLORS.get(record.levelname, '')
                if color:
                    message = f"{color}{message}{self.RESET}"
            
            return message
    
    # Configure the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler with colored formatter
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    
    # Set the colored formatter
    formatter = ColoredFormatter(
        '%(levelname)s: %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)


def main():
    # Setup colored logging first
    setup_colored_logging()
    
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
    parser.add_argument("--disable-streaming", 
                       action="store_true",
                       help="Disable streaming responses (get complete response at once)")
    parser.add_argument("--max-history", 
                       type=int,
                       default=5,
                       help="Maximum number of conversation turns to remember (default: 5)")
    parser.add_argument("--max-tokens", 
                       type=int,
                       default=2048,
                       help="Maximum number of tokens to generate (default: 2048)")
    parser.add_argument("--rerank-method",
                       choices=["cross-encoder", "cosine"],
                       default="cross-encoder",
                       help="Re-ranking method: cross-encoder (default, slower but more accurate) or cosine (faster, uses cosine similarity)")
    
    args = parser.parse_args()
    
    # Initialize the RAG system
    rag = MemoryAlphaRAG(
        chroma_db_path=args.chroma_db,
        ollama_url=args.ollama_url,
        model=args.model,
        thinking_mode=ThinkingMode(args.thinking_mode),
        show_performance=args.show_performance,
        enable_streaming=not args.disable_streaming,
        max_history_turns=args.max_history,
        rerank_method=args.rerank_method
    )
    
    if args.question:
        # Single question mode
        print(f"QUERY: {args.question}")
        response = rag.answer_question(args.question, max_tokens=args.max_tokens)
        print()  # Extra line after response
    else:
        # Interactive chat mode
        rag.chat_loop(default_max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
