import logging
import os
from typing import List


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _parse_list(value: str | None, default: List[str]) -> List[str]:
    if value is None or not value.strip():
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


class AppSettings:
    def __init__(self) -> None:
        self.app_env = os.getenv("APP_ENV", "dev")
        self.log_level = os.getenv("LOG_LEVEL", "INFO")

        self.cors_allow_origins = _parse_list(
            os.getenv("CORS_ALLOW_ORIGINS"),
            ["http://localhost:8000", "http://127.0.0.1:8000"],
        )

        self.require_api_key = _parse_bool(os.getenv("REQUIRE_API_KEY"), False)
        self.api_key = os.getenv("API_KEY", "")
        self.api_key_header_name = os.getenv("API_KEY_HEADER", "x-api-key")

        self.default_top_k = _parse_int(os.getenv("DEFAULT_TOP_K"), 10)
        self.max_top_k = _parse_int(os.getenv("MAX_TOP_K"), 50)
        self.max_query_length = _parse_int(os.getenv("MAX_QUERY_LENGTH"), 256)

        # Simple in-process rate limits for single-instance deployments.
        self.rate_limit_window_seconds = _parse_int(
            os.getenv("RATE_LIMIT_WINDOW_SECONDS"),
            60,
        )
        self.rate_limit_max_requests = _parse_int(
            os.getenv("RATE_LIMIT_MAX_REQUESTS"),
            120,
        )

        # Startup behavior
        self.strict_model_loading = _parse_bool(
            os.getenv("STRICT_MODEL_LOADING"),
            False,
        )


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
