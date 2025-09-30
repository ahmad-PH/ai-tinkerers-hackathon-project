#!/usr/bin/env python3
"""
Example usage of the Literature Review Agent with ReAct architecture.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.literature_review.literature_review import LiteratureReviewAgent

def main():
    """Demonstrate the literature review agent."""
    
    print("📚 Literature Review Agent Demo")
    print("=" * 50)
    
    # Initialize the agent
    print("🚀 Initializing agent...")
    agent = LiteratureReviewAgent()
    
    # Example queries
    queries = [
        "reinforcement learning from human feedback"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        print("-" * 50)
        
        try:
            result = agent.literature_review(query)
            print(f"\n📖 Results for '{query}':")
            print(result)
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
        
        print("\n" + "="*50)
        
        # Ask user if they want to continue
        if i < len(queries):
            response = input("\nPress Enter to continue to next query, or 'q' to quit: ")
            if response.lower() == 'q':
                break
    
    print("\n✅ Demo completed!")

if __name__ == "__main__":
    main()
