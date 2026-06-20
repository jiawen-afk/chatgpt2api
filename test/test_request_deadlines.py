from __future__ import annotations

import base64
import unittest
from unittest import mock

from services.openai_backend_api import ChatRequirements, OpenAIBackendAPI
from services.protocol.conversation import ConversationRequest, stream_text_deltas


PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


class RequestDeadlineTests(unittest.TestCase):
    def test_text_stream_passes_deadline_to_sse_reader(self) -> None:
        backend = OpenAIBackendAPI.__new__(OpenAIBackendAPI)
        backend.base_url = "https://chatgpt.test"
        backend.session = mock.Mock()
        response = mock.Mock(status_code=200)
        response._stream_closed = False
        response.close = mock.Mock()
        backend.session.post.return_value = response
        backend._bootstrap = mock.Mock()
        backend._get_chat_requirements = mock.Mock(return_value=ChatRequirements(token="requirements"))
        backend._chat_target = mock.Mock(return_value=("/backend-api/conversation", "Asia/Shanghai"))
        backend._conversation_payload = mock.Mock(return_value={"action": "next"})
        backend._conversation_headers = mock.Mock(return_value={})

        with (
            mock.patch("services.openai_backend_api.time.monotonic", return_value=100.0),
            mock.patch("services.openai_backend_api.iter_sse_payloads", return_value=iter(["[DONE]"])) as iter_payloads,
        ):
            list(backend.stream_conversation(prompt="hello", model="gpt-5"))

        self.assertEqual(backend.session.post.call_args.kwargs["timeout"], 300.0)
        self.assertEqual(iter_payloads.call_args.kwargs["deadline"], 400.0)
        response.close.assert_called_once()
        self.assertFalse(hasattr(backend, "request_deadline"))

    def test_text_retry_backend_inherits_original_request_deadline(self) -> None:
        parent_backend = OpenAIBackendAPI.__new__(OpenAIBackendAPI)
        parent_backend.access_token = "token-1"
        created_backends: list[OpenAIBackendAPI] = []

        def new_backend(access_token: str = "") -> OpenAIBackendAPI:
            backend = OpenAIBackendAPI.__new__(OpenAIBackendAPI)
            backend.access_token = access_token
            created_backends.append(backend)
            return backend

        def fake_events(active_backend: OpenAIBackendAPI, **_kwargs):
            if active_backend.access_token == "token-1":
                raise RuntimeError("token_invalidated")
            yield {"type": "conversation.delta", "delta": "ok"}

        with (
            mock.patch("services.protocol.conversation.time.monotonic", return_value=100.0),
            mock.patch("services.protocol.conversation.OpenAIBackendAPI", side_effect=new_backend),
            mock.patch("services.protocol.conversation.conversation_events", side_effect=fake_events),
            mock.patch("services.protocol.conversation.account_service.refresh_access_token", return_value="token-2"),
            mock.patch("services.protocol.conversation.account_service.mark_text_used"),
        ):
            chunks = list(stream_text_deltas(parent_backend, ConversationRequest(prompt="hello", model="gpt-5")))

        self.assertEqual(chunks, ["ok"])
        self.assertEqual([getattr(backend, "request_deadline", None) for backend in created_backends], [400.0, 400.0])

    def test_shared_bootstrap_and_token_requests_use_request_deadline_when_present(self) -> None:
        backend = OpenAIBackendAPI.__new__(OpenAIBackendAPI)
        backend.base_url = "https://chatgpt.test"
        backend.access_token = "token"
        backend.user_agent = "ua"
        backend.request_deadline = 105.0
        backend.pow_script_sources = []
        backend.pow_data_build = ""
        backend.session = mock.Mock()
        backend._headers = mock.Mock(return_value={})
        backend._bootstrap_headers = mock.Mock(return_value={})
        bootstrap_response = mock.Mock(status_code=200, text="")
        prepare_response = mock.Mock(status_code=200)
        prepare_response.json.return_value = {"prepare_token": "prepare"}
        finalize_response = mock.Mock(status_code=200)
        finalize_response.json.return_value = {"token": "requirements"}
        backend.session.get.return_value = bootstrap_response
        backend.session.post.side_effect = [prepare_response, finalize_response]

        with mock.patch("services.openai_backend_api.time.monotonic", return_value=100.0):
            backend._bootstrap()
            requirements = backend._get_chat_requirements()

        self.assertEqual(requirements.token, "requirements")
        self.assertEqual(backend.session.get.call_args.kwargs["timeout"], 5.0)
        self.assertEqual([call.kwargs["timeout"] for call in backend.session.post.call_args_list], [5.0, 5.0])

    def test_search_sets_request_deadline_for_prepare_run_and_wait(self) -> None:
        backend = OpenAIBackendAPI.__new__(OpenAIBackendAPI)
        backend.access_token = "token"
        seen_deadlines: list[float | None] = []

        def record_prepare(*_args):
            seen_deadlines.append(getattr(backend, "request_deadline", None))
            return "conduit"

        def record_run(*_args):
            seen_deadlines.append(getattr(backend, "request_deadline", None))
            return "conversation"

        def record_wait(*_args):
            seen_deadlines.append(getattr(backend, "request_deadline", None))
            return {"answer": "done"}

        backend._prepare_search_conversation = mock.Mock(side_effect=record_prepare)
        backend._bootstrap = mock.Mock()
        backend._run_search_conversation = mock.Mock(side_effect=record_run)
        backend._wait_search_result = mock.Mock(side_effect=record_wait)

        with mock.patch("services.openai_backend_api.time.monotonic", return_value=100.0):
            result = backend.search("query", timeout_secs=5.0)

        self.assertEqual(result, {"answer": "done"})
        self.assertEqual(seen_deadlines, [105.0, 105.0, 105.0])
        self.assertFalse(hasattr(backend, "request_deadline"))

    def test_search_stream_and_poll_requests_use_remaining_deadline(self) -> None:
        backend = OpenAIBackendAPI.__new__(OpenAIBackendAPI)
        backend.base_url = "https://chatgpt.test"
        backend.request_deadline = 105.0
        backend.session = mock.Mock()
        response = mock.Mock(status_code=200)
        response._stream_closed = False
        response.close = mock.Mock()
        backend.session.post.return_value = response
        backend._get_chat_requirements = mock.Mock(return_value=ChatRequirements(token="requirements"))
        backend._image_headers = mock.Mock(return_value={})

        with (
            mock.patch("services.openai_backend_api.time.monotonic", return_value=100.0),
            mock.patch("services.openai_backend_api.iter_sse_payloads", return_value=iter(['{"conversation_id":"conv"}'])) as iter_payloads,
        ):
            conversation_id = backend._run_search_conversation("query", "conduit", "gpt-5-search-api")

        self.assertEqual(conversation_id, "conv")
        self.assertEqual(backend.session.post.call_args.kwargs["timeout"], 5.0)
        self.assertEqual(iter_payloads.call_args.kwargs["deadline"], 105.0)

        backend._headers = mock.Mock(return_value={})
        backend.session.get.return_value = mock.Mock(status_code=200)
        backend.session.get.return_value.json.return_value = {}
        with mock.patch("services.openai_backend_api.time.monotonic", return_value=100.0):
            backend._get_search_conversation("conv")

        self.assertEqual(backend.session.get.call_args.kwargs["timeout"], 5.0)

    def test_editable_upload_and_stream_requests_use_remaining_deadline(self) -> None:
        backend = OpenAIBackendAPI.__new__(OpenAIBackendAPI)
        backend.base_url = "https://chatgpt.test"
        backend.user_agent = "ua"
        backend.request_deadline = 105.0
        backend.session = mock.Mock()
        backend._headers = mock.Mock(return_value={})
        create_response = mock.Mock(status_code=200)
        create_response.json.return_value = {
            "file_id": "file-1",
            "library_file_id": "library-1",
            "upload_url": "https://upload.test/blob",
        }
        uploaded_response = mock.Mock(status_code=200)
        put_response = mock.Mock(status_code=200)
        backend.session.post.side_effect = [create_response, uploaded_response]
        backend.session.put.return_value = put_response

        with mock.patch("services.openai_backend_api.time.monotonic", return_value=100.0):
            meta = backend._upload_editable_base64_image(
                f"data:image/png;base64,{base64.b64encode(PNG_BYTES).decode('ascii')}",
                1,
            )

        self.assertEqual(meta["file_id"], "file-1")
        self.assertEqual([call.kwargs["timeout"] for call in backend.session.post.call_args_list], [5.0, 5.0])
        self.assertEqual(backend.session.put.call_args.kwargs["timeout"], 5.0)

        backend.session.post.reset_mock()
        backend.session.post.side_effect = None
        stream_response = mock.Mock(status_code=200)
        stream_response._stream_closed = False
        stream_response.close = mock.Mock()
        backend.session.post.return_value = stream_response
        backend._bootstrap = mock.Mock()
        backend._get_chat_requirements = mock.Mock(return_value=ChatRequirements(token="requirements"))
        backend._image_headers = mock.Mock(return_value={})

        with (
            mock.patch("services.openai_backend_api.time.monotonic", return_value=100.0),
            mock.patch("services.openai_backend_api.iter_sse_payloads", return_value=iter(['{"conversation_id":"conv"}'])) as iter_payloads,
        ):
            conversation_id = backend._run_editable_conversation("make ppt", [meta], "conduit")

        self.assertEqual(conversation_id, "conv")
        self.assertEqual(backend.session.post.call_args.kwargs["timeout"], 5.0)
        self.assertEqual(iter_payloads.call_args.kwargs["deadline"], 105.0)

    def test_editable_poll_and_download_requests_use_remaining_deadline(self) -> None:
        backend = OpenAIBackendAPI.__new__(OpenAIBackendAPI)
        backend.base_url = "https://chatgpt.test"
        backend.request_deadline = 105.0
        backend.session = mock.Mock()
        backend._editable_conversation_document_headers = mock.Mock(return_value={})
        backend._editable_download_headers = mock.Mock(return_value={})
        backend._editable_browser_headers = mock.Mock(return_value={})

        detail_response = mock.Mock(status_code=200)
        detail_response.json.return_value = {}
        backend.session.get.return_value = detail_response
        with mock.patch("services.openai_backend_api.time.monotonic", return_value=100.0):
            backend._get_editable_conversation_detail("conv")

        self.assertEqual(backend.session.get.call_args.kwargs["timeout"], 5.0)

        artifact = mock.Mock()
        artifact.attachment_id = "file-1"
        artifact.file_id = ""
        artifact.sandbox_path = ""
        artifact.message_id = ""
        artifact.name = "deck.pptx"
        artifact.mime_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        download_url_response = mock.Mock(status_code=200)
        download_url_response.json.return_value = {"download_url": "https://download.test/deck.pptx"}
        backend.session.get.return_value = download_url_response
        with mock.patch("services.openai_backend_api.time.monotonic", return_value=100.0):
            download_url = backend._resolve_editable_download_url("conv", artifact)

        self.assertEqual(download_url, "https://download.test/deck.pptx")
        self.assertEqual(backend.session.get.call_args.kwargs["timeout"], 5.0)


if __name__ == "__main__":
    unittest.main()
