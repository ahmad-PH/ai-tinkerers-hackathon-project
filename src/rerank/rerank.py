from typing import List, Optional
from dataclasses import dataclass

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

def rerank(query: str, papers: List[str], user_feedbacks=Optional[List[bool]]):
    return papers
