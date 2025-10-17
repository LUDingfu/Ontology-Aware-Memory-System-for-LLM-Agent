import pytest
from unittest.mock import patch, MagicMock
from sqlmodel import Session
from app.services.memory_service import MemoryService
from app.models.memory import Memory, ChatEvent


class TestMemoryService:
    """Test cases for the MemoryService."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        return MagicMock(spec=Session)

    def test_memory_service_initialization(self, mock_session):
        """Test that MemoryService initializes correctly."""
        service = MemoryService(mock_session)
        assert service.session == mock_session

    def test_store_memory_success(self, mock_session):
        """Test successful memory storage."""
        with patch('app.services.memory_service.EmbeddingService') as mock_embedding:
            mock_embedding.return_value.generate_embedding.return_value = [0.1] * 1536
            
            service = MemoryService(mock_session)
            memory_data = {
                "session_id": "test-session",
                "kind": "episodic",
                "text": "User asked about order status",
                "importance": 0.8
            }
            
            result = service.store_memory(mock_session, memory_data)
            
            assert result is not None
            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()

    def test_store_memory_without_embedding(self, mock_session):
        """Test memory storage without generating embedding."""
        service = MemoryService()
        memory_data = {
            "session_id": "test-session",
            "kind": "episodic",
            "text": "User asked about order status",
            "importance": 0.8,
            "embedding": [0.1] * 1536
        }
        
        result = service.store_memory(mock_session, memory_data)
        
        assert result is not None
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    def test_get_memories_success(self, mock_session):
        """Test successful memory retrieval."""
        mock_memory = Memory(
            memory_id=1,
            session_id="test-session",
            kind="episodic",
            text="Test memory",
            embedding=[0.1] * 1536,
            importance=0.8
        )
        mock_session.exec.return_value.all.return_value = [mock_memory]
        
        service = MemoryService()
        result = service.get_user_memories(
            mock_session,
            user_id="test-user",
            limit=10,
            kind="episodic",
            importance_threshold=0.5
        )
        
        assert len(result) == 1
        assert result[0].text == "Test memory"
        mock_session.exec.assert_called_once()

    def test_get_memories_empty_result(self, mock_session):
        """Test memory retrieval with empty result."""
        mock_session.exec.return_value.all.return_value = []
        
        service = MemoryService()
        result = service.get_user_memories(
            mock_session,
            user_id="test-user"
        )
        
        assert len(result) == 0
        mock_session.exec.assert_called_once()

    def test_search_similar_memories_success(self, mock_session):
        """Test successful similar memory search."""
        mock_memory = Memory(
            memory_id=1,
            session_id="test-session",
            kind="episodic",
            text="Similar memory",
            embedding=[0.1] * 1536,
            importance=0.8
        )
        mock_session.exec.return_value.all.return_value = [mock_memory]
        
        service = MemoryService()
        query_embedding = [0.1] * 1536
        result = service.search_similar_memories(
            mock_session,
            query_embedding,
            user_id="test-user",
            limit=5
        )
        
        assert len(result) == 1
        assert result[0].text == "Similar memory"
        mock_session.exec.assert_called_once()

    def test_consolidate_memories_success(self, mock_session):
        """Test successful memory consolidation."""
        mock_memories = [
            Memory(
                memory_id=1,
                session_id="test-session",
                kind="episodic",
                text="Memory 1",
                embedding=[0.1] * 1536,
                importance=0.6
            ),
            Memory(
                memory_id=2,
                session_id="test-session",
                kind="episodic",
                text="Memory 2",
                embedding=[0.2] * 1536,
                importance=0.7
            )
        ]
        mock_session.exec.return_value.all.return_value = mock_memories
        
        service = MemoryService()
        result = service.consolidate_memories(
            mock_session,
            user_id="test-user",
            threshold=0.5
        )
        
        assert result >= 0  # Should return number of consolidated memories
        mock_session.exec.assert_called()

    def test_consolidate_memories_no_candidates(self, mock_session):
        """Test memory consolidation with no candidates."""
        mock_session.exec.return_value.all.return_value = []
        
        service = MemoryService()
        result = service.consolidate_memories(
            mock_session,
            user_id="test-user",
            threshold=0.5
        )
        
        assert result == 0
        mock_session.exec.assert_called()

    def test_store_chat_event_success(self, mock_session):
        """Test successful chat event storage."""
        service = MemoryService()
        event_data = {
            "session_id": "test-session",
            "role": "user",
            "content": "Hello, how are you?"
        }
        
        result = service.store_chat_event(mock_session, event_data)
        
        assert result is not None
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    def test_store_chat_event_invalid_role(self, mock_session):
        """Test chat event storage with invalid role."""
        service = MemoryService()
        event_data = {
            "session_id": "test-session",
            "role": "invalid_role",
            "content": "Hello"
        }
        
        with pytest.raises(ValueError, match="Invalid role"):
            service.store_chat_event(mock_session, event_data)

    def test_get_memory_summary_success(self, mock_session):
        """Test successful memory summary generation."""
        mock_memories = [
            Memory(
                memory_id=1,
                session_id="test-session",
                kind="episodic",
                text="User asked about order status",
                embedding=[0.1] * 1536,
                importance=0.8
            ),
            Memory(
                memory_id=2,
                session_id="test-session",
                kind="semantic",
                text="User prefers email communication",
                embedding=[0.2] * 1536,
                importance=0.9
            )
        ]
        mock_session.exec.return_value.all.return_value = mock_memories
        
        service = MemoryService()
        result = service.get_memory_summary(
            mock_session,
            user_id="test-user"
        )
        
        assert result is not None
        assert "summary" in result
        assert "memory_count" in result
        assert result["memory_count"] == 2

    def test_get_memory_summary_empty_memories(self, mock_session):
        """Test memory summary with no memories."""
        mock_session.exec.return_value.all.return_value = []
        
        service = MemoryService()
        result = service.get_memory_summary(
            mock_session,
            user_id="test-user"
        )
        
        assert result is not None
        assert result["memory_count"] == 0
        assert result["summary"] == "No memories found for this user."

    def test_calculate_memory_importance(self):
        """Test memory importance calculation."""
        service = MemoryService()
        
        # Test with different text lengths and content
        short_text = "Hi"
        long_text = "This is a very long text that should have higher importance because it contains more information and context about the user's request."
        
        importance_short = service.calculate_memory_importance(short_text)
        importance_long = service.calculate_memory_importance(long_text)
        
        assert 0.0 <= importance_short <= 1.0
        assert 0.0 <= importance_long <= 1.0
        assert importance_long > importance_short  # Longer text should be more important
