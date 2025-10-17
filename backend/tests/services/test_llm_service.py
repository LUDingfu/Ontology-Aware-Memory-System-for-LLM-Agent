import pytest
from unittest.mock import patch, MagicMock
from app.services.llm_service import LLMService


class TestLLMService:
    """Test cases for the LLMService."""

    def test_llm_service_initialization(self):
        """Test that LLMService initializes correctly."""
        with patch('app.services.llm_service.OpenAI'):
            service = LLMService()
            assert service.model == "gpt-4"
            assert service.max_tokens == 2000
            assert service.temperature == 0.7

    def test_generate_response_success(self):
        """Test successful response generation."""
        mock_response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "This is a test response."
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
        
        with patch('app.services.llm_service.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            service = LLMService()
            messages = [{"role": "user", "content": "Hello"}]
            result = service.generate_response(messages)
            
            assert result == "This is a test response."
            mock_client.chat.completions.create.assert_called_once()

    def test_generate_response_with_context(self):
        """Test response generation with context."""
        mock_response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Based on the context, here's my response."
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 30,
                "total_tokens": 130
            }
        }
        
        with patch('app.services.llm_service.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            service = LLMService()
            messages = [{"role": "user", "content": "Tell me about John"}]
            context = {"memories": ["John is a customer"], "domain_facts": []}
            
            result = service.generate_response(messages, context)
            
            assert result == "Based on the context, here's my response."
            mock_client.chat.completions.create.assert_called_once()

    def test_generate_response_api_error(self):
        """Test response generation when OpenAI API fails."""
        with patch('app.services.llm_service.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = Exception("API error")
            mock_openai.return_value = mock_client
            
            service = LLMService()
            messages = [{"role": "user", "content": "Hello"}]
            
            with pytest.raises(Exception, match="API error"):
                service.generate_response(messages)

    def test_generate_response_empty_messages(self):
        """Test response generation with empty messages."""
        with patch('app.services.llm_service.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            
            service = LLMService()
            
            with pytest.raises(ValueError, match="Messages cannot be empty"):
                service.generate_response([])

    def test_generate_response_none_messages(self):
        """Test response generation with None messages."""
        with patch('app.services.llm_service.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            
            service = LLMService()
            
            with pytest.raises(ValueError, match="Messages cannot be None"):
                service.generate_response(None)

    def test_generate_response_invalid_message_format(self):
        """Test response generation with invalid message format."""
        with patch('app.services.llm_service.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            
            service = LLMService()
            messages = [{"invalid": "format"}]
            
            with pytest.raises(ValueError, match="Invalid message format"):
                service.generate_response(messages)

    def test_generate_response_max_tokens_exceeded(self):
        """Test response generation when max tokens is exceeded."""
        mock_response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "This response is too long and gets truncated..."
                    },
                    "finish_reason": "length"
                }
            ],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 2000,
                "total_tokens": 2050
            }
        }
        
        with patch('app.services.llm_service.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            service = LLMService()
            messages = [{"role": "user", "content": "Generate a very long response"}]
            
            result = service.generate_response(messages)
            
            assert result == "This response is too long and gets truncated..."
            # Should still work, but might be truncated

    def test_generate_response_with_system_message(self):
        """Test response generation with system message."""
        mock_response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "I understand the system instructions."
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 60,
                "completion_tokens": 25,
                "total_tokens": 85
            }
        }
        
        with patch('app.services.llm_service.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            service = LLMService()
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"}
            ]
            
            result = service.generate_response(messages)
            
            assert result == "I understand the system instructions."
            mock_client.chat.completions.create.assert_called_once()
