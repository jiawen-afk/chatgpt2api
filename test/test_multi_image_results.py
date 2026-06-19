from __future__ import annotations

import base64
import queue
import unittest
from unittest import mock

from curl_cffi.requests.models import STREAM_END

from services.config import config
from services.openai_backend_api import ImagePollTimeoutError, OpenAIBackendAPI
from services.protocol.conversation import (
    ConversationRequest,
    ImageOutput,
    _generate_single_image,
    extract_conversation_ids,
    is_tls_connection_error,
    stream_image_outputs,
)
from services.protocol.openai_v1_response import stream_image_response
from utils.helper import iter_sse_payloads


PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAFgwJ/lm2C6wAAAABJRU5ErkJggg=="
)


def _conversation(file_ids: list[str], sediment_ids: list[str] | None = None) -> dict:
    parts: list[object] = [
        {"content_type": "image_asset_pointer", "asset_pointer": f"file-service://{file_id}"}
        for file_id in file_ids
    ]
    parts.extend(f"sediment://{sediment_id}" for sediment_id in (sediment_ids or []))
    return {
        "mapping": {
            "tool": {
                "message": {
                    "author": {"role": "tool"},
                    "create_time": 1,
                    "metadata": {"async_task_type": "image_gen"},
                    "content": {"content_type": "multimodal_text", "parts": parts},
                }
            }
        }
    }


class FakeBackend(OpenAIBackendAPI):
    def __init__(self, conversations: list[dict] | None = None) -> None:
        self.conversations = conversations or []
        self.calls = 0
        self.file_urls: dict[str, str] = {}
        self.sediment_urls: dict[str, str] = {}

    def _get_conversation(self, conversation_id: str, timeout_secs: float = 60.0) -> dict:
        self.calls += 1
        index = min(self.calls - 1, len(self.conversations) - 1)
        return self.conversations[index]

    def _get_file_download_url(self, file_id: str, timeout_secs: float = 60.0) -> str:
        return self.file_urls.get(file_id, "")

    def _get_attachment_download_url(
        self,
        conversation_id: str,
        attachment_id: str,
        timeout_secs: float = 60.0,
    ) -> str:
        return self.sediment_urls.get(attachment_id, "")


class MultiImageResultTests(unittest.TestCase):
    def test_stream_id_extractor_keeps_full_file_ids(self) -> None:
        payload = (
            '{"conversation_id":"conv-1"} '
            'file-service://file-first_123-extra sediment://sed-second_456-extra'
        )

        conversation_id, file_ids, sediment_ids = extract_conversation_ids(payload)

        self.assertEqual(conversation_id, "conv-1")
        self.assertEqual(file_ids, ["file-first_123-extra"])
        self.assertEqual(sediment_ids, ["sed-second_456-extra"])

    def test_http2_internal_error_is_treated_as_retryable_stream_error(self) -> None:
        message = "curl: (92) HTTP/2 stream 1 was not closed cleanly: INTERNAL_ERROR (err 2)"

        self.assertTrue(is_tls_connection_error(message))

    def test_conversation_record_extractor_finds_all_generated_assets(self) -> None:
        backend = FakeBackend()
        conversation = {
            "mapping": {
                "user": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["file-service://file-user-input"]},
                    }
                },
                "tool": {
                    "message": {
                        "author": {"role": "tool"},
                        "create_time": 1,
                        "metadata": {
                            "async_task_type": "image_gen",
                            "nested": {"asset": "file-service://file-second"},
                        },
                        "content": {
                            "content_type": "text",
                            "parts": [
                                {"content_type": "image_asset_pointer", "asset_pointer": "file-service://file-first"},
                                "sediment://sed-first",
                            ],
                        },
                    }
                },
                "assistant": {
                    "message": {
                        "author": {"role": "assistant"},
                        "create_time": 2,
                        "metadata": {},
                        "content": {
                            "parts": [
                                {"content_type": "image_asset_pointer", "asset_pointer": "file-service://file-third"}
                            ]
                        },
                    }
                },
            }
        }

        records = backend._extract_image_tool_records(conversation)
        file_ids = [file_id for record in records for file_id in record["file_ids"]]
        sediment_ids = [sediment_id for record in records for sediment_id in record["sediment_ids"]]

        self.assertEqual(file_ids, ["file-first", "file-second", "file-third"])
        self.assertEqual(sediment_ids, ["sed-first"])

    def test_poll_waits_for_generated_asset_ids_to_settle(self) -> None:
        backend = FakeBackend([
            _conversation(["file-one"]),
            _conversation(["file-one", "file-two"], ["sed-one"]),
            _conversation(["file-one", "file-two"], ["sed-one"]),
        ])

        with (
            mock.patch.dict(config.data, {
                "image_poll_initial_wait_secs": 0,
                "image_poll_interval_secs": 0.5,
                "image_check_before_hit_enabled": True,
                "image_settle_enabled": True,
            }),
            mock.patch("services.openai_backend_api.time.sleep", lambda _seconds: None),
        ):
            file_ids, sediment_ids = backend._poll_image_results("conv-1", timeout_secs=10)

        self.assertEqual(file_ids, ["file-one", "file-two"])
        self.assertEqual(sediment_ids, ["sed-one"])
        self.assertEqual(backend.calls, 3)

    def test_resolver_uses_file_and_sediment_urls(self) -> None:
        backend = FakeBackend()
        backend.file_urls = {"file-one": "https://files.test/one.png"}
        backend.sediment_urls = {
            "sed-one": "https://attachments.test/one.png",
            "sed-two": "https://attachments.test/two.png",
        }

        urls = backend._resolve_image_urls("conv-1", ["file-one"], ["sed-one", "sed-two"])

        self.assertEqual(urls, [
            "https://files.test/one.png",
            "https://attachments.test/one.png",
            "https://attachments.test/two.png",
        ])

    def test_resolver_keeps_stream_ids_when_poll_extension_fails(self) -> None:
        backend = FakeBackend()
        backend.file_urls = {"file-one": "https://files.test/one.png"}
        backend._get_conversation = mock.Mock(side_effect=RuntimeError("poll failed"))

        with mock.patch("services.openai_backend_api.time.sleep", lambda _seconds: None):
            urls = backend.resolve_conversation_image_urls("conv-1", ["file-one"], [], poll=True)

        self.assertEqual(urls, ["https://files.test/one.png"])

    def test_responses_stream_emits_all_image_output_items(self) -> None:
        first = base64.b64encode(b"first").decode("ascii")
        second = base64.b64encode(b"second").decode("ascii")
        events = list(stream_image_response(
            [ImageOutput(
                kind="result",
                model="gpt-image-2",
                index=1,
                total=1,
                data=[{"b64_json": first}, {"b64_json": second}],
            )],
            "draw two options",
            "gpt-image-2",
        ))

        done_events = [event for event in events if event.get("type") == "response.output_item.done"]
        completed = next(event["response"] for event in events if event.get("type") == "response.completed")

        self.assertEqual([event["output_index"] for event in done_events], [0, 1])
        self.assertEqual([item["result"] for item in completed["output"]], [first, second])

    def test_text_reply_retry_poll_uses_configured_timeout_without_300_second_floor(self) -> None:
        backend = mock.Mock()
        backend.stream_conversation.return_value = iter([
            '{"conversation_id":"conv-1","message":{"author":{"role":"assistant"},'
            '"content":{"parts":["{\\"referenced_image_ids\\":[\\"file-input\\"]}"]}}}',
            "[DONE]",
        ])
        backend.resolve_conversation_image_urls.return_value = []
        backend._poll_image_results.side_effect = RuntimeError("temporary upstream failure")

        with (
            mock.patch.dict(config.data, {"image_poll_timeout_secs": 7, "image_poll_initial_wait_secs": 0}),
            mock.patch("services.protocol.conversation.time.sleep", lambda _seconds: None),
        ):
            outputs = list(stream_image_outputs(
                backend,
                ConversationRequest(prompt="draw", model="gpt-image-2"),
            ))

        self.assertEqual(outputs[-1].kind, "message")
        self.assertEqual([call.args[1] for call in backend._poll_image_results.call_args_list], [7, 7, 7])

    def test_image_generation_sse_request_uses_configured_image_timeout(self) -> None:
        backend = OpenAIBackendAPI.__new__(OpenAIBackendAPI)
        backend.base_url = "https://chatgpt.test"
        backend.session = mock.Mock()
        backend.session.post.return_value.status_code = 200
        backend._image_headers = mock.Mock(return_value={"Accept": "text/event-stream"})
        requirements = mock.Mock(token="token", proof_token="", turnstile_token="", so_token="")

        with mock.patch.dict(config.data, {"image_poll_timeout_secs": 11}):
            backend._start_image_generation("draw", requirements, "conduit", "gpt-image-2")

        self.assertEqual(backend.session.post.call_args.kwargs["timeout"], 11)

    def test_image_prepare_request_uses_remaining_deadline(self) -> None:
        backend = OpenAIBackendAPI.__new__(OpenAIBackendAPI)
        backend.base_url = "https://chatgpt.test"
        backend.image_request_deadline = 105.0
        backend.session = mock.Mock()
        backend.session.post.return_value.status_code = 200
        backend.session.post.return_value.json.return_value = {"conduit_token": "conduit"}
        backend._image_headers = mock.Mock(return_value={})
        requirements = mock.Mock(token="token", proof_token="", turnstile_token="", so_token="")

        with mock.patch("services.openai_backend_api.time.monotonic", return_value=100.0):
            token = backend._prepare_image_conversation("draw", requirements, "gpt-image-2")

        self.assertEqual(token, "conduit")
        self.assertEqual(backend.session.post.call_args.kwargs["timeout"], 5.0)

    def test_image_upload_requests_use_remaining_deadline(self) -> None:
        backend = OpenAIBackendAPI.__new__(OpenAIBackendAPI)
        backend.base_url = "https://chatgpt.test"
        backend.user_agent = "ua"
        backend.image_request_deadline = 105.0
        backend.session = mock.Mock()
        backend._headers = mock.Mock(return_value={})
        create_response = mock.Mock(status_code=200)
        create_response.json.return_value = {"file_id": "file-1", "upload_url": "https://upload.test/blob"}
        uploaded_response = mock.Mock(status_code=200)
        put_response = mock.Mock(status_code=200)
        backend.session.post.side_effect = [create_response, uploaded_response]
        backend.session.put.return_value = put_response

        with mock.patch("services.openai_backend_api.time.monotonic", return_value=100.0):
            meta = backend._upload_image(f"data:image/png;base64,{base64.b64encode(PNG_BYTES).decode('ascii')}")

        self.assertEqual(meta["file_id"], "file-1")
        self.assertEqual([call.kwargs["timeout"] for call in backend.session.post.call_args_list], [5.0, 5.0])
        self.assertEqual(backend.session.put.call_args.kwargs["timeout"], 5.0)

    def test_chat_requirements_keep_default_timeouts_without_image_deadline(self) -> None:
        backend = OpenAIBackendAPI.__new__(OpenAIBackendAPI)
        backend.base_url = "https://chatgpt.test"
        backend.access_token = "token"
        backend.user_agent = "ua"
        backend.pow_script_sources = []
        backend.pow_data_build = ""
        backend.session = mock.Mock()
        backend._headers = mock.Mock(return_value={})
        prepare_response = mock.Mock(status_code=200)
        prepare_response.json.return_value = {"prepare_token": "prepare"}
        finalize_response = mock.Mock(status_code=200)
        finalize_response.json.return_value = {"token": "requirements"}
        backend.session.post.side_effect = [prepare_response, finalize_response]

        with mock.patch.dict(config.data, {"image_poll_timeout_secs": 7}):
            requirements = backend._get_chat_requirements()

        self.assertEqual(requirements.token, "requirements")
        self.assertEqual([call.kwargs["timeout"] for call in backend.session.post.call_args_list], [30.0, 30.0])

    def test_poll_image_results_caps_conversation_fetch_to_remaining_budget(self) -> None:
        backend = OpenAIBackendAPI.__new__(OpenAIBackendAPI)
        backend.base_url = "https://chatgpt.test"
        backend.session = mock.Mock()
        backend._headers = mock.Mock(return_value={})
        backend._query_backend_tasks = mock.Mock(return_value=[])
        response = mock.Mock(status_code=200)
        response.json.return_value = _conversation(["file-result"])
        backend.session.get.return_value = response

        with (
            mock.patch.dict(config.data, {
                "image_poll_initial_wait_secs": 0,
                "image_check_before_hit_enabled": False,
            }),
            mock.patch("services.openai_backend_api.time.time", side_effect=[100.0, 100.0, 101.0, 101.0]),
        ):
            file_ids, sediment_ids = backend._poll_image_results("conv-1", timeout_secs=5.0)

        self.assertEqual(file_ids, ["file-result"])
        self.assertEqual(sediment_ids, [])
        self.assertEqual(backend.session.get.call_args.kwargs["timeout"], 4.0)

    def test_image_download_uses_remaining_deadline(self) -> None:
        backend = OpenAIBackendAPI.__new__(OpenAIBackendAPI)
        backend.image_request_deadline = 105.0
        backend.session = mock.Mock()
        response = mock.Mock(status_code=200, content=b"image")
        backend.session.get.return_value = response

        with mock.patch("services.openai_backend_api.time.monotonic", return_value=100.0):
            images = backend.download_image_bytes(["https://download.test/image.png"])

        self.assertEqual(images, [b"image"])
        self.assertEqual(backend.session.get.call_args.kwargs["timeout"], 5.0)

    def test_picture_conversation_passes_deadline_to_sse_payload_iterator(self) -> None:
        backend = OpenAIBackendAPI.__new__(OpenAIBackendAPI)
        backend.access_token = "token"
        backend.image_request_deadline = 123.0
        backend.progress_callback = None
        backend._upload_image = mock.Mock()
        backend._bootstrap = mock.Mock()
        backend._get_chat_requirements = mock.Mock(return_value=mock.Mock(token="token", proof_token="", turnstile_token="", so_token=""))
        backend._prepare_image_conversation = mock.Mock(return_value="conduit")
        response = mock.Mock()
        response._stream_closed = False
        response.close = mock.Mock()
        backend._start_image_generation = mock.Mock(return_value=response)

        with mock.patch("services.openai_backend_api.iter_sse_payloads", return_value=iter(["[DONE]"])) as iter_payloads:
            list(backend._stream_picture_conversation("draw", "gpt-image-2", []))

        self.assertEqual(iter_payloads.call_args.kwargs["deadline"], 123.0)
        self.assertIn("ChatGPT 生图超时", iter_payloads.call_args.kwargs["timeout_message"])
        response.close.assert_called_once()

    def test_image_poll_timeout_retry_stops_when_configured_budget_is_exhausted(self) -> None:
        backend = mock.Mock()
        backend.image_request_deadline = None
        token = "token-1"

        def fail_poll(*_args, **_kwargs):
            raise ImagePollTimeoutError("timed out")

        with (
            mock.patch.dict(config.data, {"image_poll_timeout_secs": 1}),
            mock.patch("services.protocol.conversation.time.monotonic", side_effect=[100.0, 100.1, 101.2]),
            mock.patch("services.protocol.conversation.account_service.get_available_access_token", return_value=token) as get_token,
            mock.patch("services.protocol.conversation.account_service.get_account", return_value={"email": "a@example.test"}),
            mock.patch("services.protocol.conversation.account_service.mark_image_result"),
            mock.patch("services.protocol.conversation.OpenAIBackendAPI", return_value=backend),
            mock.patch("services.protocol.conversation.stream_image_outputs", side_effect=fail_poll),
        ):
            with self.assertRaises(ImagePollTimeoutError):
                _generate_single_image(ConversationRequest(prompt="draw", model="gpt-image-2"), 1, 1)

        self.assertEqual(get_token.call_count, 1)
        self.assertEqual(backend.image_request_deadline, 101.0)

    def test_sse_payload_iterator_enforces_application_deadline_for_idle_curl_stream(self) -> None:
        response = mock.Mock()
        response.queue = queue.Queue()
        response.quit_now = mock.Mock()
        response.quit_now.set = mock.Mock()
        response._stream_closed = False
        response.iter_lines.side_effect = AssertionError("deadline path should not use blocking iter_lines")

        with mock.patch("utils.helper.time.monotonic", side_effect=[100.0, 100.6]):
            with self.assertRaises(TimeoutError):
                list(iter_sse_payloads(response, deadline=100.5, timeout_message="image stream timed out"))

        response.quit_now.set.assert_called_once()
        self.assertTrue(response._stream_closed)

    def test_sse_payload_iterator_preserves_line_parsing_with_deadline_queue(self) -> None:
        response = mock.Mock()
        response.queue = queue.Queue()
        response.quit_now = mock.Mock()
        response._stream_closed = False
        response.iter_lines.side_effect = AssertionError("deadline path should not use blocking iter_lines")
        response.queue.put(b"data: one\n\ndata: t")
        response.queue.put(b"wo\n\n")
        response.queue.put(STREAM_END)

        with mock.patch("utils.helper.time.monotonic", side_effect=[100.0, 100.0, 100.0]):
            payloads = list(iter_sse_payloads(response, deadline=101.0))

        self.assertEqual(payloads, ["one", "two"])


if __name__ == "__main__":
    unittest.main()
