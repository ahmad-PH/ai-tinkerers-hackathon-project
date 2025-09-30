import arxiv
from typing import List

def search_arxiv(query: str, max_results: int = 5) -> List[arxiv.Result]:
    """
    Search ArXiv for papers matching the given query.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
    
    Returns:
        List of arxiv.Result objects
    """
    # Create a search query
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    # Execute the search and return results
    results = list(search.results())
    return results