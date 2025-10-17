from collections.abc import Generator
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, create_engine, SQLModel
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.main import app


@pytest.fixture(scope="session")
def db() -> Generator[Session, None, None]:
    """Create a test database session."""
    # Create an in-memory SQLite database for testing
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    
    with Session(engine) as session:
        yield session


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """Create a test client."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="function")
def session_id() -> str:
    """Generate a unique session ID for testing."""
    return str(uuid4())


@pytest.fixture(scope="function")
def test_user_id() -> str:
    """Generate a test user ID."""
    return "test-user-123"


@pytest.fixture(scope="function")
def mock_openai_embedding():
    """Mock OpenAI embedding response."""
    return {
        "data": [
            {
                "embedding": [0.1] * 1536,  # Mock embedding vector
                "index": 0
            }
        ],
        "model": "text-embedding-3-small",
        "usage": {
            "prompt_tokens": 10,
            "total_tokens": 10
        }
    }


@pytest.fixture(scope="function")
def mock_openai_chat():
    """Mock OpenAI chat completion response."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a test response from the AI assistant."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 20,
            "total_tokens": 70
        }
    }