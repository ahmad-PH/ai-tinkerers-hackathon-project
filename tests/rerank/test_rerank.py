import pytest
from src.rerank.rerank import rerank, Paper


class TestRerank:
    """Test cases for the rerank function."""
    
    def setup_method(self):
        """Set up sample papers for testing."""
        self.sample_papers = [
            Paper(
                title="Attention Is All You Need",
                authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit", "Llion Jones", "Aidan N. Gomez", "Łukasz Kaiser", "Illia Polosukhin"],
                abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
                pdf_url="https://arxiv.org/pdf/1706.03762.pdf",
                entry_id="1706.03762"
            ),
            Paper(
                title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
                authors=["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"],
                abstract="We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.",
                pdf_url="https://arxiv.org/pdf/1810.04805.pdf",
                entry_id="1810.04805"
            ),
            Paper(
                title="ResNet: Deep Residual Learning for Image Recognition",
                authors=["Kaiming He", "Xiangyu Zhang", "Shaoqing Ren", "Jian Sun"],
                abstract="Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions.",
                pdf_url="https://arxiv.org/pdf/1512.03385.pdf",
                entry_id="1512.03385"
            ),
            Paper(
                title="Generative Adversarial Networks",
                authors=["Ian Goodfellow", "Jean Pouget-Abadie", "Mehdi Mirza", "Bing Xu", "David Warde-Farley", "Sherjil Ozair", "Aaron Courville", "Yoshua Bengio"],
                abstract="We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G.",
                pdf_url="https://arxiv.org/pdf/1406.2661.pdf",
                entry_id="1406.2661"
            ),
            Paper(
                title="Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
                authors=["Nitish Srivastava", "Geoffrey Hinton", "Alex Krizhevsky", "Ilya Sutskever", "Ruslan Salakhutdinov"],
                abstract="Deep neural nets with a large number of parameters are very powerful machine learning systems. However, overfitting is a serious problem in such networks. Large networks are also slow to use, making it difficult to deal with overfitting by combining the predictions of many different large neural nets at test time.",
                pdf_url="https://arxiv.org/pdf/1207.0580.pdf",
                entry_id="1207.0580"
            )
        ]
    
    def test_rerank_input_output_size_match(self):
        """Test that rerank returns the same number of papers as input."""
        query = "transformer architecture"
        paper_titles = [paper.title for paper in self.sample_papers]
        user_feedbacks = [True, False, True, False, True]
        
        result = rerank(query, paper_titles, user_feedbacks)
        
        assert len(result) == len(paper_titles), f"Expected {len(paper_titles)} papers, got {len(result)}"
    
    def test_rerank_with_empty_input(self):
        """Test rerank with empty input."""
        query = "machine learning"
        paper_titles = []
        user_feedbacks = []
        
        result = rerank(query, paper_titles, user_feedbacks)
        
        assert len(result) == 0, f"Expected empty result, got {len(result)} papers"
    
    def test_rerank_with_single_paper(self):
        """Test rerank with single paper."""
        query = "neural networks"
        paper_titles = [self.sample_papers[0].title]
        user_feedbacks = [True]
        
        result = rerank(query, paper_titles, user_feedbacks)
        
        assert len(result) == 1, f"Expected 1 paper, got {len(result)}"
        assert result[0] == paper_titles[0], "Single paper should remain unchanged"
    
    def test_rerank_with_different_query_types(self):
        """Test rerank with different types of queries."""
        queries = [
            "transformer",
            "deep learning",
            "computer vision",
            "natural language processing"
        ]
        
        paper_titles = [paper.title for paper in self.sample_papers]
        user_feedbacks = [True, False, True, False, True]
        
        for query in queries:
            result = rerank(query, paper_titles, user_feedbacks)
            assert len(result) == len(paper_titles), f"Query '{query}' should return same number of papers"
    
    def test_rerank_with_varying_feedback_lengths(self):
        """Test rerank with different feedback array lengths."""
        query = "machine learning"
        paper_titles = [paper.title for paper in self.sample_papers]
        
        # Test with fewer feedbacks than papers
        user_feedbacks_short = [True, False]
        result_short = rerank(query, paper_titles, user_feedbacks_short)
        assert len(result_short) == len(paper_titles), "Should handle fewer feedbacks than papers"
        
        # Test with more feedbacks than papers
        user_feedbacks_long = [True, False, True, False, True, False, True]
        result_long = rerank(query, paper_titles, user_feedbacks_long)
        assert len(result_long) == len(paper_titles), "Should handle more feedbacks than papers"
    
    def test_rerank_preserves_paper_content(self):
        """Test that rerank preserves the actual paper content (not just count)."""
        query = "attention mechanism"
        paper_titles = [paper.title for paper in self.sample_papers]
        user_feedbacks = [True, False, True, False, True]
        
        result = rerank(query, paper_titles, user_feedbacks)
        
        # Check that all original papers are still in the result
        for original_title in paper_titles:
            assert original_title in result, f"Original paper '{original_title}' should be in result"
        
        # Check that no new papers were added
        for result_title in result:
            assert result_title in paper_titles, f"Result paper '{result_title}' should be from original list"
    
    def test_rerank_with_none_feedbacks(self):
        """Test rerank with None feedbacks."""
        query = "deep learning"
        paper_titles = [paper.title for paper in self.sample_papers]
        user_feedbacks = [None, None, None, None, None]
        
        result = rerank(query, paper_titles, user_feedbacks)
        
        assert len(result) == len(paper_titles), "Should handle None feedbacks"
    
    def test_rerank_with_mixed_feedback_types(self):
        """Test rerank with mixed feedback types."""
        query = "neural networks"
        paper_titles = [paper.title for paper in self.sample_papers]
        user_feedbacks = [True, False, None, True, False]
        
        result = rerank(query, paper_titles, user_feedbacks)
        
        assert len(result) == len(paper_titles), "Should handle mixed feedback types"


if __name__ == "__main__":
    pytest.main([__file__])