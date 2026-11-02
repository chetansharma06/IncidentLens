import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from fastapi.testclient import TestClient
from google.genai.errors import ClientError
from langchain_core.exceptions import OutputParserException
from langchain_core.documents import Document

from app import app
from src.config import Settings, get_settings
from src.gemini_errors import GeminiConfigurationError, gemini_error_response
from src.self_rag import _format_context, _llm, grade_relevance
from src.vectorstore import ensure_index, get_embeddings


class GeminiConfigurationTests(unittest.TestCase):
    def tearDown(self):
        get_settings.cache_clear()
        get_embeddings.cache_clear()

    def test_default_chat_model_is_current_gemini_flash(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()

        self.assertEqual(settings.gemini_model, "gemini-3.8-flash")

    def test_settings_read_gemini_configuration_from_environment(self):
        with patch.dict(
            os.environ,
            {
                "GEMINI_API_KEY": "test-key",
                "GEMINI_MODEL": "gemini-test-model",
                "EMBEDDING_MODEL": "gemini-test-embedding",
                "EMBEDDING_DIMENSION": "768",
            },
            clear=True,
        ):
            settings = Settings()

        self.assertEqual(settings.gemini_api_key, "test-key")
        self.assertEqual(settings.gemini_model, "gemini-test-model")
        self.assertEqual(settings.embedding_model, "gemini-test-embedding")
        self.assertEqual(settings.embedding_dimension, 768)

    def test_llm_requires_a_gemini_api_key(self):
        with patch("src.self_rag.get_settings", return_value=Settings(gemini_api_key="")):
            with self.assertRaisesRegex(GeminiConfigurationError, "GEMINI_API_KEY"):
                _llm()

    def test_api_key_and_provider_failures_map_to_safe_responses(self):
        cases = [
            (
                GeminiConfigurationError("missing key"),
                503,
                "Gemini is not configured. Set GEMINI_API_KEY on the server.",
            ),
            (
                ClientError(403, {"message": "sensitive provider details"}),
                502,
                "Gemini authentication or model access failed. Verify the server-side API key and access.",
            ),
            (
                ClientError(429, {"message": "sensitive provider details"}),
                429,
                "Gemini rate limit or quota reached. Try again later or check your AI Studio limits.",
            ),
            (
                ClientError(503, {"message": "sensitive provider details"}),
                503,
                "Gemini is temporarily unavailable. Try again shortly.",
            ),
            (
                ClientError(400, {"message": "sensitive provider details"}),
                502,
                "Gemini rejected the generated request. Check the configured model and request parameters.",
            ),
            (
                httpx.ConnectError("sensitive network details"),
                503,
                "Could not reach Gemini. Check the network connection and try again.",
            ),
            (
                OutputParserException("sensitive response details"),
                502,
                "Gemini returned an unexpected response. Please retry the request.",
            ),
        ]

        for error, expected_status, expected_message in cases:
            with self.subTest(status=expected_status):
                mapped = gemini_error_response(error)
                self.assertEqual(mapped, (expected_status, expected_message))

    def test_llm_uses_configured_gemini_model(self):
        settings = Settings(gemini_api_key="test-key", gemini_model="gemini-test-model")
        with patch("src.self_rag.get_settings", return_value=settings):
            with patch("src.self_rag.ChatGoogleGenerativeAI") as llm:
                _llm()

        llm.assert_called_once_with(
            google_api_key="test-key",
            model="gemini-test-model",
            temperature=0,
        )

    def test_embedding_factory_uses_gemini_configuration(self):
        settings = Settings(
            gemini_api_key="test-key",
            embedding_model="gemini-test-embedding",
            embedding_dimension=768,
        )
        with patch("src.vectorstore.get_settings", return_value=settings):
            with patch("src.vectorstore.GoogleGenerativeAIEmbeddings") as embeddings:
                get_embeddings()

        embeddings.assert_called_once_with(
            google_api_key="test-key",
            model="gemini-test-embedding",
            output_dimensionality=768,
        )

    def test_index_creation_uses_embedding_dimension(self):
        settings = Settings(
            pinecone_api_key="test-pinecone-key",
            pinecone_index_name="test-gemini-index",
            embedding_dimension=768,
        )
        client = Mock()
        client.list_indexes.return_value = []
        index = client.Index.return_value
        with patch("src.vectorstore.get_settings", return_value=settings):
            with patch("src.vectorstore.get_pinecone_client", return_value=client):
                with patch("src.vectorstore.ServerlessSpec", return_value="spec"):
                    result = ensure_index()

        self.assertIs(result, index)
        client.create_index.assert_called_once_with(
            name="test-gemini-index",
            dimension=768,
            metric="cosine",
            spec="spec",
        )

    def test_index_dimension_mismatch_fails_clearly(self):
        settings = Settings(
            pinecone_api_key="test-pinecone-key",
            pinecone_index_name="test-gemini-index",
            embedding_dimension=3072,
        )
        client = Mock()
        client.list_indexes.return_value = [SimpleNamespace(name="test-gemini-index")]
        client.describe_index.return_value = {"dimension": 768}
        with patch("src.vectorstore.get_settings", return_value=settings):
            with patch("src.vectorstore.get_pinecone_client", return_value=client):
                with self.assertRaisesRegex(RuntimeError, "dimension 768"):
                    ensure_index()

    def test_context_format_preserves_source_labels(self):
        context = _format_context(
            [
                Document(
                    page_content="Rollback procedure",
                    metadata={"source": "rollback.md", "page": 0},
                ),
                Document(
                    page_content="External guidance",
                    metadata={
                        "source_type": "web",
                        "title": "Provider guide",
                        "url": "https://example.test/guide",
                    },
                ),
            ]
        )

        self.assertIn("[INTERNAL 1] rollback.md | page 1", context)
        self.assertIn("[WEB 2] Provider guide | https://example.test/guide", context)

    def test_relevance_grading_surfaces_provider_errors(self):
        llm = Mock()
        llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("provider unavailable")
        state = {
            "question": "Why did the checkout API fail?",
            "docs": [Document(page_content="Runbook context")],
        }
        with patch("src.self_rag._llm", return_value=llm):
            with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
                grade_relevance(state)

    def test_health_endpoint_starts_without_external_services(self):
        with patch("app.init_db"):
            with TestClient(app) as client:
                response = client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_chat_endpoint_maps_gemini_quota_error_without_exposing_provider_body(self):
        with patch("app.init_db"):
            with patch("app.run_self_rag", side_effect=ClientError(429, {"message": "sensitive provider details"})):
                with TestClient(app) as client:
                    response = client.post(
                        "/api/chat",
                        json={"question": "Why did checkout fail?", "thread_id": "test-thread"},
                    )

        self.assertEqual(response.status_code, 429)
        self.assertIn("quota", response.json()["detail"])
        self.assertNotIn("sensitive provider details", response.text)

    def test_chat_endpoint_reports_missing_gemini_configuration_safely(self):
        with patch("app.init_db"):
            with patch("app.run_self_rag", side_effect=GeminiConfigurationError("missing key")):
                with TestClient(app) as client:
                    response = client.post(
                        "/api/chat",
                        json={"question": "Why did checkout fail?", "thread_id": "test-thread"},
                    )

        self.assertEqual(response.status_code, 503)
        self.assertIn("GEMINI_API_KEY", response.json()["detail"])

    def test_unexpected_error_does_not_expose_exception_details(self):
        with patch("app.init_db"):
            with patch("app.run_self_rag", side_effect=RuntimeError("sensitive internal details")):
                with TestClient(app) as client:
                    response = client.post(
                        "/api/chat",
                        json={"question": "Why did checkout fail?", "thread_id": "test-thread"},
                    )

        self.assertEqual(response.status_code, 500)
        self.assertNotIn("sensitive internal details", response.text)


if __name__ == "__main__":
    unittest.main()