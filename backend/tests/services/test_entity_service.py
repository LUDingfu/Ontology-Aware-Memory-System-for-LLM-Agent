import pytest
from unittest.mock import patch, MagicMock
from sqlmodel import Session
from app.services.entity_service import EntityService
from app.models.memory import Entity


class TestEntityService:
    """Test cases for the EntityService."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        return MagicMock(spec=Session)

    def test_entity_service_initialization(self, mock_session):
        """Test that EntityService initializes correctly."""
        service = EntityService(mock_session)
        assert service.session == mock_session

    def test_extract_entities_success(self):
        """Test successful entity extraction."""
        service = EntityService(mock_session)
        text = "I want to check the status of order #12345 for customer John Smith"
        
        result = service.extract_entities(text)
        
        assert isinstance(result, list)
        # Should extract order number and customer name
        assert len(result) > 0

    def test_extract_entities_empty_text(self):
        """Test entity extraction with empty text."""
        service = EntityService(mock_session)
        
        result = service.extract_entities("")
        
        assert result == []

    def test_extract_entities_none_text(self):
        """Test entity extraction with None text."""
        service = EntityService(mock_session)
        
        result = service.extract_entities(None)
        
        assert result == []

    def test_extract_entities_no_entities(self):
        """Test entity extraction with text containing no entities."""
        service = EntityService(mock_session)
        text = "This is just a regular sentence with no specific entities."
        
        result = service.extract_entities(text)
        
        assert isinstance(result, list)
        # May or may not find entities depending on implementation

    def test_link_entities_success(self, mock_session):
        """Test successful entity linking."""
        mock_customer = MagicMock()
        mock_customer.customer_id = 123
        mock_customer.name = "John Smith"
        mock_customer.email = "john@example.com"
        
        mock_session.exec.return_value.first.return_value = mock_customer
        
        service = EntityService(mock_session)
        entities = [
            {"name": "John Smith", "type": "customer", "confidence": 0.9}
        ]
        
        result = service.link_entities(mock_session, entities)
        
        assert len(result) == 1
        assert result[0]["external_ref"]["customer_id"] == 123
        assert result[0]["source"] == "db"

    def test_link_entities_no_match(self, mock_session):
        """Test entity linking with no database matches."""
        mock_session.exec.return_value.first.return_value = None
        
        service = EntityService(mock_session)
        entities = [
            {"name": "Unknown Person", "type": "customer", "confidence": 0.9}
        ]
        
        result = service.link_entities(mock_session, entities)
        
        assert len(result) == 1
        assert result[0]["external_ref"] is None
        assert result[0]["source"] == "message"

    def test_link_entities_multiple_types(self, mock_session):
        """Test entity linking with multiple entity types."""
        mock_order = MagicMock()
        mock_order.order_id = 456
        mock_order.order_number = "ORD-12345"
        
        mock_session.exec.return_value.first.side_effect = [None, mock_order]
        
        service = EntityService(mock_session)
        entities = [
            {"name": "Unknown Customer", "type": "customer", "confidence": 0.9},
            {"name": "ORD-12345", "type": "order", "confidence": 0.95}
        ]
        
        result = service.link_entities(mock_session, entities)
        
        assert len(result) == 2
        assert result[0]["external_ref"] is None
        assert result[1]["external_ref"]["order_id"] == 456

    def test_store_entities_success(self, mock_session):
        """Test successful entity storage."""
        service = EntityService(mock_session)
        entities = [
            {
                "session_id": "test-session",
                "name": "John Smith",
                "type": "customer",
                "source": "db",
                "external_ref": {"customer_id": 123}
            }
        ]
        
        result = service.store_entities(mock_session, entities)
        
        assert len(result) == 1
        mock_session.add.assert_called()
        mock_session.commit.assert_called_once()

    def test_store_entities_empty_list(self, mock_session):
        """Test entity storage with empty list."""
        service = EntityService(mock_session)
        
        result = service.store_entities(mock_session, [])
        
        assert result == []
        mock_session.add.assert_not_called()

    def test_get_entities_success(self, mock_session):
        """Test successful entity retrieval."""
        mock_entity = Entity(
            entity_id=1,
            session_id="test-session",
            name="John Smith",
            type="customer",
            source="db",
            external_ref={"customer_id": 123}
        )
        mock_session.exec.return_value.all.return_value = [mock_entity]
        
        service = EntityService(mock_session)
        result = service.get_entities_for_session(
            mock_session,
            session_id="test-session",
            limit=10,
            type_filter="customer",
            source_filter="db"
        )
        
        assert len(result) == 1
        assert result[0].name == "John Smith"
        mock_session.exec.assert_called_once()

    def test_get_entities_empty_result(self, mock_session):
        """Test entity retrieval with empty result."""
        mock_session.exec.return_value.all.return_value = []
        
        service = EntityService(mock_session)
        result = service.get_entities_for_session(
            mock_session,
            session_id="test-session"
        )
        
        assert len(result) == 0
        mock_session.exec.assert_called_once()

    def test_get_entities_with_filters(self, mock_session):
        """Test entity retrieval with filters."""
        mock_entity = Entity(
            entity_id=1,
            session_id="test-session",
            name="John Smith",
            type="customer",
            source="db"
        )
        mock_session.exec.return_value.all.return_value = [mock_entity]
        
        service = EntityService(mock_session)
        result = service.get_entities_for_session(
            mock_session,
            session_id="test-session",
            limit=5,
            type_filter="customer",
            source_filter="db"
        )
        
        assert len(result) == 1
        mock_session.exec.assert_called_once()

    def test_fuzzy_match_entities_success(self):
        """Test successful fuzzy entity matching."""
        service = EntityService(mock_session)
        entity_name = "John Smith"
        candidates = [
            {"name": "John Smith", "id": 1},
            {"name": "Jon Smith", "id": 2},
            {"name": "John Smyth", "id": 3},
            {"name": "Jane Doe", "id": 4}
        ]
        
        result = service.fuzzy_match_entities(entity_name, candidates)
        
        assert result is not None
        assert result["id"] == 1  # Should match exact name first

    def test_fuzzy_match_entities_no_match(self):
        """Test fuzzy entity matching with no good matches."""
        service = EntityService(mock_session)
        entity_name = "Completely Different Name"
        candidates = [
            {"name": "John Smith", "id": 1},
            {"name": "Jane Doe", "id": 2}
        ]
        
        result = service.fuzzy_match_entities(entity_name, candidates)
        
        assert result is None  # No good match found

    def test_fuzzy_match_entities_empty_candidates(self):
        """Test fuzzy entity matching with empty candidates."""
        service = EntityService(mock_session)
        entity_name = "John Smith"
        candidates = []
        
        result = service.fuzzy_match_entities(entity_name, candidates)
        
        assert result is None

    def test_calculate_entity_confidence(self):
        """Test entity confidence calculation."""
        service = EntityService(mock_session)
        
        # Test exact match
        confidence_exact = service.calculate_entity_confidence("John Smith", "John Smith")
        assert confidence_exact == 1.0
        
        # Test partial match
        confidence_partial = service.calculate_entity_confidence("John Smith", "John")
        assert 0.0 < confidence_partial < 1.0
        
        # Test no match
        confidence_none = service.calculate_entity_confidence("John Smith", "Jane Doe")
        assert confidence_none == 0.0
