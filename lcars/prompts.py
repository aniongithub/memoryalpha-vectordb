#!/usr/bin/env python3
"""
System prompts and response formatting for the Memory Alpha RAG system
"""

from typing import List, Dict

# Handle both direct execution and module import
from thinking_mode import ThinkingMode

def get_system_prompt(thinking_mode: ThinkingMode) -> str:
    """Generate the LCARS-style system prompt based on thinking mode"""
    
    think_instruction = "Use <think> tags for analysis. " if thinking_mode != ThinkingMode.DISABLED else ""
    
    response_format = "Working...\n{think_block}[Answer]".format(
        think_block="<think>[Analysis]</think>\n" if thinking_mode != ThinkingMode.DISABLED else ""
    )
    
    return f"""LCARS computer. {think_instruction}Use records. Be precise. Single paragraph. Format: {response_format}"""

def get_user_prompt(context_text: str, query: str, conversation_history: List[Dict[str, str]] = None) -> str:
    """Format user prompt with context, query, and conversation history"""
    
    # Build conversation context if history exists
    conversation_context = ""
    if conversation_history and len(conversation_history) > 0:
        conversation_context = "\nContext:\n"
        for exchange in conversation_history[-3:]:  # Last 3 exchanges only
            conversation_context += f"Q: {exchange['question'][:40]}{'...' if len(exchange['question']) > 40 else ''}\n"
            conversation_context += f"A: {exchange['answer'][:60]}{'...' if len(exchange['answer']) > 60 else ''}\n"
    
    return f"""Records:
{context_text}{conversation_context}

Query: {query}"""
