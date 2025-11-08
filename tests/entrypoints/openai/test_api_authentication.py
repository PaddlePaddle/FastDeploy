import asyncio
import hashlib
import secrets
from unittest.mock import AsyncMock, Mock

import pytest
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from fastdeploy.entrypoints.openai.middleware import AuthenticationMiddleware


def mock_asgi_app() -> tuple[ASGIApp, Mock]:
    mock_send = Mock()

    async def mock_app(scope: Scope, receive: Receive, send: Send) -> None:
        await asyncio.sleep(0)
        mock_send(scope=scope, receive=receive, send=send)

    return mock_app, mock_send


def create_test_scope(
    path: str = "/v1/chat/completions", method: str = "POST", headers: dict = None, scope_type: str = "http"
) -> Scope:
    headers = headers or {}
    scope_headers = []
    for k, v in headers.items():
        key_bytes = str(k).lower().encode("latin-1")
        value_bytes = str(v).lower().encode("latin-1")
        scope_headers.append((key_bytes, value_bytes))
    return {
        "type": scope_type,
        "method": method,
        "headers": scope_headers,
        "path": path,
        "root_path": "",
    }


class TestAuthenticationMiddleware:
    VALID_TOKENS = ["test_key_123", "another_valid_key_456"]
    INVALID_TOKEN = "wrong_key_789"
    EXPECTED_TOKEN_HASHES = [hashlib.sha256(t.encode("utf-8")).digest() for t in VALID_TOKENS]

    @pytest.fixture(autouse=True)
    def setup_middleware(self):
        self.mock_app, self.mock_send = mock_asgi_app()
        self.middleware = AuthenticationMiddleware(app=self.mock_app, tokens=self.VALID_TOKENS)

    def test_verify_token_no_authorization_header(self):
        headers = Headers()
        assert self.middleware.verify_token(headers) is False

    def test_verify_token_invalid_scheme(self):
        headers = Headers({"Authorization": "Basic wrong_scheme"})
        assert self.middleware.verify_token(headers) is False

    def test_verify_token_valid_token(self):
        for valid_token in self.VALID_TOKENS:
            headers = Headers({"Authorization": f"Bearer {valid_token}"})
            assert self.middleware.verify_token(headers) is True

    def test_verify_token_invalid_token(self):
        headers = Headers({"Authorization": f"Bearer {self.INVALID_TOKEN}"})
        assert self.middleware.verify_token(headers) is False

    def test_verify_token_hash_comparison(self):
        valid_token = self.VALID_TOKENS[0]
        param_hash = hashlib.sha256(valid_token.encode("utf-8")).digest()

        assert self.middleware.api_tokens == self.EXPECTED_TOKEN_HASHES

        with pytest.MonkeyPatch.context() as mp:
            mock_compare = Mock(return_value=True)
            mp.setattr(secrets, "compare_digest", mock_compare)

            headers = Headers({"Authorization": f"Bearer {valid_token}"})
            self.middleware.verify_token(headers)
            assert mock_compare.call_count == len(self.EXPECTED_TOKEN_HASHES)
            mock_compare.assert_any_call(param_hash, self.EXPECTED_TOKEN_HASHES[0])

    @pytest.mark.asyncio
    async def test_call_skip_non_v1_path(self):
        for path in ["/health", "/metrics", "/docs"]:
            scope = create_test_scope(path=path)
            receive = AsyncMock()
            send = AsyncMock()

            await self.middleware(scope, receive, send)

            self.mock_send.assert_called_once_with(scope=scope, receive=receive, send=send)
            self.mock_send.reset_mock()

    @pytest.mark.asyncio
    async def test_call_skip_options_method(self):
        scope = create_test_scope(method="OPTIONS", path="/v1/chat/completions")
        receive = AsyncMock()
        send = AsyncMock()

        await self.middleware(scope, receive, send)

        self.mock_send.assert_called_once_with(scope=scope, receive=receive, send=send)

    @pytest.mark.asyncio
    async def test_call_skip_non_http_websocket_scope(self):
        for scope_type in ["lifespan", "startup", "shutdown"]:
            scope = create_test_scope(scope_type=scope_type)
            receive = AsyncMock()
            send = AsyncMock()

            await self.middleware(scope, receive, send)

            self.mock_send.assert_called_once_with(scope=scope, receive=receive, send=send)
            self.mock_send.reset_mock()

    @pytest.mark.asyncio
    async def test_call_v1_path_valid_token(self):
        scope = create_test_scope(headers={"Authorization": f"Bearer {self.VALID_TOKENS[0]}"})
        print(scope)
        receive = AsyncMock()
        send = AsyncMock()

        headers = Headers(scope=scope)
        assert self.middleware.verify_token(headers) is True
        await self.middleware(scope, receive, send)

        self.mock_send.assert_called_once_with(scope=scope, receive=receive, send=send)

    @pytest.mark.asyncio
    async def test_call_v1_path_invalid_token(self):
        scope = create_test_scope(headers={"Authorization": f"Bearer {self.INVALID_TOKEN}"})
        receive = AsyncMock()
        send = AsyncMock()

        await self.middleware(scope, receive, send)

        self.mock_send.assert_not_called()
        assert send.called
        send_call = send.call_args[0][0]
        assert isinstance(send_call, dict)
        assert "Unauthorized" in send_call["body"].decode("utf-8")

    @pytest.mark.asyncio
    async def test_call_v1_path_no_token(self):
        scope = create_test_scope(headers={})
        receive = AsyncMock()
        send = AsyncMock()

        await self.middleware(scope, receive, send)

        self.mock_send.assert_not_called()
        send_call = send.call_args[0][0]
        assert isinstance(send_call, dict)
        assert "Unauthorized" in send_call["body"].decode("utf-8")
