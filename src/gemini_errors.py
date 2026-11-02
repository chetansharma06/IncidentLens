import httpx
from google.genai.errors import APIError, UnknownApiResponseError
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError


class GeminiConfigurationError(RuntimeError):
    pass


def gemini_error_response(error: Exception) -> tuple[int, str] | None:
    if isinstance(error, GeminiConfigurationError):
        return 503, "Gemini is not configured. Set GEMINI_API_KEY on the server."

    if isinstance(error, APIError):
        if error.code in {401, 403}:
            return 502, "Gemini authentication or model access failed. Verify the server-side API key and access."
        if error.code == 429:
            return 429, "Gemini rate limit or quota reached. Try again later or check your AI Studio limits."
        if error.code >= 500:
            return 503, "Gemini is temporarily unavailable. Try again shortly."
        if error.code == 400:
            return 502, "Gemini rejected the generated request. Check the configured model and request parameters."
        return 502, "Gemini could not complete the request."

    if isinstance(error, (httpx.TimeoutException, httpx.NetworkError)):
        return 503, "Could not reach Gemini. Check the network connection and try again."

    if isinstance(error, (UnknownApiResponseError, OutputParserException, ValidationError)):
        return 502, "Gemini returned an unexpected response. Please retry the request."

    return None
