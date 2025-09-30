from typing import List, Optional
from dataclasses import dataclass
import numpy as np
import os
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env file
load_dotenv()

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


def get_embedding(text: str) -> np.ndarray:
    """Get embedding for a text using Gemini embedding model."""
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
        return np.array(embedding_vector)
        
    except Exception as e:
        print(f"Error getting embedding: {e}")
        # Return a zero vector as fallback
        return np.zeros(768)  # Gemini text-embedding-004 has 768 dimensions


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
    
    print("Papers and their scores, each on new line: ")
    for paper, score in paper_scores:
        print(f"{paper} - {score}")

    # Return papers in order of relevance
    return [paper for paper, score in paper_scores]
