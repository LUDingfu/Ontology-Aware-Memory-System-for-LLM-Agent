import pytest
from unittest.mock import patch, MagicMock
from sqlmodel import Session
from app.services.retrieval_service import RetrievalService


class TestRetrievalService:
    """Test cases for the RetrievalService."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        return MagicMock(spec=Session)

    def test_retrieval_service_initialization(self, mock_session):
        """Test that RetrievalService initializes correctly."""
        service = RetrievalService(mock_session)
        assert service.session == mock_session

    def test_retrieve_context_success(self, mock_session):
        """Test successful context retrieval."""
        with patch('app.services.retrieval_service.MemoryService') as mock_memory, \
             patch('app.services.retrieval_service.EmbeddingService') as mock_embedding:
            
            mock_memory.return_value.search_similar_memories.return_value = [
                MagicMock(text="Relevant memory", importance=0.8)
            ]
            mock_embedding.return_value.generate_embedding.return_value = [0.1] * 1536
            
            service = RetrievalService(mock_session)
            query = "Tell me about customer John Smith"
            user_id = "test-user"
            
            result = service.retrieve_context(mock_session, query, user_id)
            
            assert "memories" in result
            assert "domain_facts" in result
            assert len(result["memories"]) == 1
            mock_memory.return_value.search_similar_memories.assert_called_once()

    def test_retrieve_context_empty_query(self, mock_session):
        """Test context retrieval with empty query."""
        service = RetrievalService()
        
        result = service.retrieve_context(mock_session, "", "test-user")
        
        assert result["memories"] == []
        assert result["domain_facts"] == []

    def test_retrieve_context_none_query(self, mock_session):
        """Test context retrieval with None query."""
        service = RetrievalService()
        
        result = service.retrieve_context(mock_session, None, "test-user")
        
        assert result["memories"] == []
        assert result["domain_facts"] == []

    def test_get_domain_facts_success(self, mock_session):
        """Test successful domain facts retrieval."""
        mock_customer = MagicMock()
        mock_customer.customer_id = 123
        mock_customer.name = "John Smith"
        mock_customer.email = "john@example.com"
        
        mock_session.exec.return_value.all.return_value = [mock_customer]
        
        service = RetrievalService()
        entities = [{"name": "John Smith", "type": "customer"}]
        
        result = service._retrieve_domain_facts(entities)
        
        assert len(result) == 1
        assert result[0]["table"] == "customers"
        assert result[0]["record_id"] == 123
        mock_session.exec.assert_called()

    def test_get_domain_facts_no_entities(self, mock_session):
        """Test domain facts retrieval with no entities."""
        service = RetrievalService()
        
        result = service.get_domain_facts(mock_session, [])
        
        assert result == []

    def test_get_domain_facts_empty_result(self, mock_session):
        """Test domain facts retrieval with empty database result."""
        mock_session.exec.return_value.all.return_value = []
        
        service = RetrievalService()
        entities = [{"name": "Unknown Customer", "type": "customer"}]
        
        result = service._retrieve_domain_facts(entities)
        
        assert result == []
        mock_session.exec.assert_called()

    def test_get_domain_facts_multiple_entity_types(self, mock_session):
        """Test domain facts retrieval with multiple entity types."""
        mock_customer = MagicMock()
        mock_customer.customer_id = 123
        mock_customer.name = "John Smith"
        
        mock_order = MagicMock()
        mock_order.order_id = 456
        mock_order.order_number = "ORD-12345"
        
        mock_session.exec.return_value.all.side_effect = [[mock_customer], [mock_order]]
        
        service = RetrievalService()
        entities = [
            {"name": "John Smith", "type": "customer"},
            {"name": "ORD-12345", "type": "order"}
        ]
        
        result = service._retrieve_domain_facts(entities)
        
        assert len(result) == 2
        assert result[0]["table"] == "customers"
        assert result[1]["table"] == "orders"

    def test_hybrid_search_success(self, mock_session):
        """Test successful hybrid search."""
        with patch('app.services.retrieval_service.MemoryService') as mock_memory, \
             patch('app.services.retrieval_service.EmbeddingService') as mock_embedding, \
             patch('app.services.retrieval_service.EntityService') as mock_entity:
            
            mock_memory.return_value.search_similar_memories.return_value = [
                MagicMock(text="Relevant memory", importance=0.8)
            ]
            mock_embedding.return_value.generate_embedding.return_value = [0.1] * 1536
            mock_entity.return_value.extract_entities.return_value = [
                {"name": "John Smith", "type": "customer", "confidence": 0.9}
            ]
            
            service = RetrievalService(mock_session)
            query = "Tell me about customer John Smith"
            user_id = "test-user"
            
            result = service.hybrid_search(mock_session, query, user_id)
            
            assert "memories" in result
            assert "domain_facts" in result
            assert "entities" in result
            assert len(result["memories"]) == 1
            assert len(result["entities"]) == 1

    def test_hybrid_search_vector_only(self, mock_session):
        """Test hybrid search with vector search only."""
        with patch('app.services.retrieval_service.MemoryService') as mock_memory, \
             patch('app.services.retrieval_service.EmbeddingService') as mock_embedding:
            
            mock_memory.return_value.search_similar_memories.return_value = [
                MagicMock(text="Relevant memory", importance=0.8)
            ]
            mock_embedding.return_value.generate_embedding.return_value = [0.1] * 1536
            
            service = RetrievalService(mock_session)
            query = "Generic question without specific entities"
            user_id = "test-user"
            
            result = service.hybrid_search(mock_session, query, user_id, vector_only=True)
            
            assert "memories" in result
            assert "domain_facts" in result
            assert len(result["memories"]) == 1
            assert len(result["domain_facts"]) == 0

    def test_hybrid_search_keyword_only(self, mock_session):
        """Test hybrid search with keyword search only."""
        with patch('app.services.retrieval_service.EntityService') as mock_entity:
            mock_entity.return_value.extract_entities.return_value = [
                {"name": "John Smith", "type": "customer", "confidence": 0.9}
            ]
            
            service = RetrievalService(mock_session)
            query = "Tell me about customer John Smith"
            user_id = "test-user"
            
            result = service.hybrid_search(mock_session, query, user_id, keyword_only=True)
            
            assert "memories" in result
            assert "domain_facts" in result
            assert "entities" in result
            assert len(result["memories"]) == 0
            assert len(result["entities"]) == 1

    def test_hybrid_search_no_results(self, mock_session):
        """Test hybrid search with no results."""
        with patch('app.services.retrieval_service.MemoryService') as mock_memory, \
             patch('app.services.retrieval_service.EmbeddingService') as mock_embedding, \
             patch('app.services.retrieval_service.EntityService') as mock_entity:
            
            mock_memory.return_value.search_similar_memories.return_value = []
            mock_embedding.return_value.generate_embedding.return_value = [0.1] * 1536
            mock_entity.return_value.extract_entities.return_value = []
            
            service = RetrievalService(mock_session)
            query = "Completely unrelated query"
            user_id = "test-user"
            
            result = service.hybrid_search(mock_session, query, user_id)
            
            assert "memories" in result
            assert "domain_facts" in result
            assert "entities" in result
            assert len(result["memories"]) == 0
            assert len(result["domain_facts"]) == 0
            assert len(result["entities"]) == 0

    def test_hybrid_search_database_error(self, mock_session):
        """Test hybrid search when database fails."""
        with patch('app.services.retrieval_service.MemoryService') as mock_memory, \
             patch('app.services.retrieval_service.EmbeddingService') as mock_embedding:
            
            mock_memory.return_value.search_similar_memories.side_effect = Exception("Database error")
            mock_embedding.return_value.generate_embedding.return_value = [0.1] * 1536
            
            service = RetrievalService(mock_session)
            query = "Test query"
            user_id = "test-user"
            
            with pytest.raises(Exception, match="Database error"):
                service.hybrid_search(mock_session, query, user_id)

    def test_hybrid_search_embedding_error(self, mock_session):
        """Test hybrid search when embedding generation fails."""
        with patch('app.services.retrieval_service.EmbeddingService') as mock_embedding:
            mock_embedding.return_value.generate_embedding.side_effect = Exception("Embedding error")
            
            service = RetrievalService(mock_session)
            query = "Test query"
            user_id = "test-user"
            
            with pytest.raises(Exception, match="Embedding error"):
                service.hybrid_search(mock_session, query, user_id)
