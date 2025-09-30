#!/usr/bin/env python3
"""
Simple test script for the literature review agent.
"""

import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from literature_review.literature_review import literature_review

def test_literature_review():
    """Test the literature review function with a simple query."""
    
    print("🧪 Testing Literature Review Agent")
    print("=" * 50)
    
    # Test query
    query = "transformer architecture attention mechanisms"
    
    print(f"Query: {query}")
    print("\n🔍 Conducting literature review...")
    print("-" * 50)
    
    try:
        result = literature_review(query)
        print("\n📚 Literature Review Results:")
        print("=" * 50)
        print(result)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Set up your Google Cloud credentials")
        print("2. Installed the arxiv-mcp-server: uv tool install arxiv-mcp-server")
        print("3. Have internet connectivity")
        
        return False
    
    print("\n✅ Test completed successfully!")
    return True

if __name__ == "__main__":
    success = test_literature_review()
    sys.exit(0 if success else 1)
