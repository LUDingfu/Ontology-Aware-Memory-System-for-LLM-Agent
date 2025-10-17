import pytest
from unittest.mock import patch, MagicMock
from app.services.embedding_service import EmbeddingService


class TestEmbeddingService:
    """Test cases for the EmbeddingService."""

    def test_embedding_service_initialization(self):
        """Test that EmbeddingService initializes correctly."""
        with patch('app.services.embedding_service.OpenAI'):
            service = EmbeddingService()
            assert service.model == "text-embedding-3-small"
            assert service.dimensions == 1536

    def test_generate_embedding_success(self):
        """Test successful embedding generation."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = [0.1] * 1536
        
        with patch('app.services.embedding_service.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            service = EmbeddingService()
            result = service.generate_embedding("Test text")
            
            assert result == [0.1] * 1536
            mock_client.embeddings.create.assert_called_once()

    def test_generate_embedding_batch_success(self):
        """Test successful batch embedding generation."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(), MagicMock()]
        mock_response.data[0].embedding = [0.1] * 1536
        mock_response.data[1].embedding = [0.2] * 1536
        
        with patch('app.services.embedding_service.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            service = EmbeddingService()
            texts = ["Text 1", "Text 2"]
            result = service.generate_embeddings(texts)
            
            assert len(result) == 2
            assert result[0] == [0.1] * 1536
            assert result[1] == [0.2] * 1536
            mock_client.embeddings.create.assert_called_once()

    def test_generate_embedding_api_error(self):
        """Test embedding generation when OpenAI API fails."""
        with patch('app.services.embedding_service.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_client.embeddings.create.side_effect = Exception("API error")
            mock_openai.return_value = mock_client
            
            service = EmbeddingService()
            result = service.generate_embedding("Test text")
            
            assert result == []  # Should return empty list on error

    def test_generate_embedding_empty_text(self):
        """Test embedding generation with empty text."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = []
        
        with patch('app.services.embedding_service.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            service = EmbeddingService()
            result = service.generate_embedding("")
            
            assert result == []  # Empty text should return empty embedding

    def test_generate_embedding_none_text(self):
        """Test embedding generation with None text."""
        with patch('app.services.embedding_service.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_client.embeddings.create.side_effect = Exception("Input cannot be None")
            mock_openai.return_value = mock_client
            
            service = EmbeddingService()
            result = service.generate_embedding(None)
            
            assert result == []  # None text should return empty list on error

    def test_generate_embeddings_empty_list(self):
        """Test batch embedding generation with empty list."""
        with patch('app.services.embedding_service.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            
            service = EmbeddingService()
            result = service.generate_embeddings([])
            
            assert result == []
            mock_client.embeddings.create.assert_not_called()

    def test_generate_embeddings_large_batch(self):
        """Test batch embedding generation with large batch."""
        # OpenAI has limits on batch size, test that we handle it correctly
        mock_response = MagicMock()
        mock_response.data = [MagicMock() for _ in range(100)]
        for i, data in enumerate(mock_response.data):
            data.embedding = [0.1] * 1536
        
        with patch('app.services.embedding_service.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            service = EmbeddingService()
            texts = [f"Text {i}" for i in range(100)]
            result = service.generate_embeddings(texts)
            
            assert len(result) == 100
            assert all(len(embedding) == 1536 for embedding in result)
