from typing import List, Optional
from dataclasses import dataclass
import numpy as np
import os
import json
import hashlib
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env file
load_dotenv()

# Cache configuration
CACHE_FILE = "embedding_cache.json"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
CACHE_PATH = os.path.join(CACHE_DIR, CACHE_FILE)

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

@dataclass
class Paper:
    """Represents a paper from a search API (e.g., arXiv)."""
    title: str
    authors: List[str]
    abstract: str
    pdf_url: Optional[str] = None
    entry_id: Optional[str] = None
    
    def __str__(self) -> str:
        """String representation of the paper."""
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += " et al."
        return f"{self.title} by {authors_str}"
    
    def to_dict(self) -> dict:
        """Convert paper to dictionary format."""
        return {
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "pdf_url": self.pdf_url,
            "entry_id": self.entry_id,
        }

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


def _get_text_hash(text: str) -> str:
    """Generate a hash for the text to use as cache key."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def _load_cache() -> dict:
    """Load the embedding cache from JSON file."""
    try:
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, 'r') as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load cache file: {e}")
    return {}


def _save_cache(cache: dict) -> None:
    """Save the embedding cache to JSON file."""
    try:
        with open(CACHE_PATH, 'w') as f:
            json.dump(cache, f, indent=2)
    except IOError as e:
        print(f"Warning: Could not save cache file: {e}")


def get_embedding(text: str) -> np.ndarray:
    """Get embedding for a text using Gemini embedding model with local caching."""
    # Generate hash for the text
    text_hash = _get_text_hash(text)
    
    # Load cache
    cache = _load_cache()
    
    # Check if embedding exists in cache
    if text_hash in cache:
        # print(f"✅ CACHE HIT: Found embedding for text hash {text_hash[:8]}... (text preview: '{text[:50]}{'...' if len(text) > 50 else ''}')")
        return np.array(cache[text_hash])
    
    # Get embedding from API
    print(f"❌ CACHE MISS: No embedding found for text hash {text_hash[:8]}... (text preview: '{text[:50]}{'...' if len(text) > 50 else ''}')")
    print(f"🔄 Fetching from Gemini API...")
    
    try:
        # Get API key from environment
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        
        # Create client and get embedding
        client = genai.Client(api_key=api_key)
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text
        )
        
        # Extract embedding vector
        embedding_vector = response.embeddings[0].values
        embedding_array = np.array(embedding_vector)
        
        # Cache the embedding
        cache[text_hash] = embedding_vector
        _save_cache(cache)
        print(f"💾 CACHED: Saved new embedding for text hash {text_hash[:8]}... (embedding dim: {len(embedding_vector)})")
        
        return embedding_array
        
    except Exception as e:
        print(f"❌ ERROR: Failed to get embedding: {e}")
        # Return a zero vector as fallback
        return np.zeros(768)  # Gemini text-embedding-004 has 768 dimensions


def clear_embedding_cache() -> None:
    """Clear the embedding cache."""
    try:
        if os.path.exists(CACHE_PATH):
            os.remove(CACHE_PATH)
            print("Embedding cache cleared.")
        else:
            print("No cache file found to clear.")
    except IOError as e:
        print(f"Error clearing cache: {e}")


def get_cache_stats() -> dict:
    """Get statistics about the embedding cache."""
    cache = _load_cache()
    cache_size = os.path.getsize(CACHE_PATH) if os.path.exists(CACHE_PATH) else 0
    
    print(f"📊 CACHE STATISTICS:")
    print(f"   • Cached embeddings: {len(cache)}")
    print(f"   • Cache file size: {cache_size:,} bytes ({cache_size/1024:.1f} KB)")
    print(f"   • Cache file path: {CACHE_PATH}")
    
    if cache:
        print(f"   • Sample cached hashes: {list(cache.keys())[:3]}")
    
    return {
        "cached_embeddings": len(cache),
        "cache_file_size": cache_size,
        "cache_file_path": CACHE_PATH
    }


def rerank(query: str, papers: List[Paper], user_feedbacks: Optional[List[bool]] = None) -> List[Paper]:
    """
    Rerank papers based on query relevance using Gemini embeddings and cosine similarity.
    
    Args:
        query: User search query
        papers: List of Paper objects to rerank
        user_feedbacks: Optional list of boolean feedback (not used in current implementation)
    
    Returns:
        List of Paper objects sorted by relevance to the query
    """
    if not papers:
        return papers
    
    # Get embedding for the query
    query_embedding = get_embedding(query)
    
    # Calculate similarities and create paper-score pairs
    paper_scores = []
    for paper in papers:
        # Combine title and abstract for better context
        paper_text = f"{paper.title}. {paper.abstract}"
        paper_embedding = get_embedding(paper_text)
        
        similarity = cosine_similarity(query_embedding, paper_embedding)
        paper_scores.append((paper, similarity))
    
    # Sort papers by similarity score (descending)
    paper_scores.sort(key=lambda x: x[1], reverse=True)

    # Return papers in order of relevance
    return [paper for paper, score in paper_scores]
