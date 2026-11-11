from __future__ import annotations

import base64
import io
import threading
import unittest
from unittest import mock

from PIL import Image

from services import image_storage_service
from services.openai_backend_api import OpenAIBackendAPI
from services.protocol import conversation, openai_v1_image_edit, openai_v1_image_generations
from services.protocol.conversation import ConversationRequest, ImageGenerationError
from utils.helper import is_gif_bytes, normalize_image_output


def animated_gif_bytes(frames: int = 2) -> bytes:
    buffer = io.BytesIO()
    images = [Image.new("RGB", (8, 8), "red" if index % 2 == 0 else "blue") for index in range(frames)]
    images[0].save(buffer, format="GIF", save_all=True, append_images=images[1:], duration=200, loop=0)
    return buffer.getvalue()


def static_bytes(fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buffer, format=fmt)
    return buffer.getvalue()


class NormalizeImageOutputTests(unittest.TestCase):
    def test_defaults_and_bounds(self) -> None:
        self.assertEqual(normalize_image_output("gif"), ("gif", 9, 200))
        self.assertEqual(normalize_image_output(), ("png", None, None))
        self.assertEqual(normalize_image_output("PNG"), ("png", None, None))
        self.assertEqual(normalize_image_output("gif", 4, 100), ("gif", 4, 100))
        self.assertEqual(normalize_image_output("gif", "64", "2000"), ("gif", 64, 2000))

    def test_invalid_combinations_raise(self) -> None:
        for args in (
            ("jpg",),
            ("png", 9),
            ("png", None, 200),
            ("gif", 1),
            ("gif", 65),
            ("gif", True),
            ("gif", 2.5),
            ("gif", 9, 25),
            ("gif", 9, 2001),
            ("gif", 9, 0),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                normalize_image_output(*args)

    def test_is_gif_bytes_only_matches_gif_headers(self) -> None:
        self.assertTrue(is_gif_bytes(animated_gif_bytes()))
        self.assertFalse(is_gif_bytes(static_bytes("PNG")))
        self.assertFalse(is_gif_bytes(b""))


class GifStorageTests(unittest.TestCase):
    def test_gif_bytes_keep_gif_extension(self) -> None:
        service = image_storage_service.ImageStorageService.__new__(image_storage_service.ImageStorageService)
        self.assertTrue(service.make_relative_path(animated_gif_bytes()).endswith(".gif"))
        self.assertTrue(service.make_relative_path(static_bytes("PNG")).endswith(".png"))

    def test_webdav_receives_gif_content_type(self) -> None:
        client = image_storage_service.WebDAVClient.__new__(image_storage_service.WebDAVClient)
        with mock.patch.object(client, "ensure_dirs"), mock.patch.object(
            client, "remote_url", return_value="https://dav.example.test/a.gif"
        ), mock.patch.object(client, "_request") as request:
            client.put("a.gif", animated_gif_bytes(), content_type="image/gif")
        self.assertEqual(request.call_args.kwargs["headers"]["Content-Type"], "image/gif")

    def test_save_uses_detected_mime_for_webdav(self) -> None:
        service = image_storage_service.ImageStorageService.__new__(image_storage_service.ImageStorageService)
        service._index_lock = threading.Lock()
        raw = animated_gif_bytes()
        with mock.patch.object(service, "settings", return_value={}), mock.patch.object(
            service, "mode", return_value="webdav"
        ), mock.patch.object(service, "_load_index", return_value={}), mock.patch.object(
            service, "_save_index"
        ), mock.patch.object(service, "_public_url", return_value="https://example.test/a.gif"), mock.patch.object(
            image_storage_service.ImageStorageService, "make_relative_path", return_value="2026/11/07/a.gif"
        ), mock.patch.object(image_storage_service, "WebDAVClient") as webdav, mock.patch.object(
            image_storage_service.config, "cleanup_old_images"
        ):
            stored = service.save(raw, "https://example.test")

        self.assertEqual(webdav.return_value.put.call_args.kwargs["content_type"], "image/gif")
        self.assertEqual(stored.rel, "2026/11/07/a.gif")


class FormatImageResultTests(unittest.TestCase):
    def test_gif_result_reports_mime_and_animated(self) -> None:
        raw = animated_gif_bytes()
        encoded = base64.b64encode(raw).decode("ascii")
        with mock.patch.object(conversation, "save_image_bytes", return_value="https://example.test/a.gif") as save:
            result = conversation.format_image_result(
                [{"b64_json": encoded}], "ball", "b64_json", output_format="gif"
            )
        save.assert_called_once_with(raw, None)
        item = result["data"][0]
        self.assertEqual(item["mime_type"], "image/gif")
        self.assertTrue(item["animated"])

    def test_png_result_is_marked_static(self) -> None:
        encoded = base64.b64encode(static_bytes("PNG")).decode("ascii")
        with mock.patch.object(conversation, "save_image_bytes", return_value="https://example.test/a.png"):
            result = conversation.format_image_result([{"b64_json": encoded}], "ball", "url")
        item = result["data"][0]
        self.assertEqual(item["mime_type"], "image/png")
        self.assertFalse(item["animated"])

    def test_static_output_cannot_masquerade_as_animation(self) -> None:
        for fmt in ("PNG", "GIF"):
            encoded = base64.b64encode(static_bytes(fmt)).decode("ascii")
            with self.subTest(fmt=fmt), mock.patch.object(
                conversation, "save_image_bytes", return_value="https://example.test/a.gif"
            ), self.assertRaises(ImageGenerationError):
                conversation.format_image_result(
                    [{"b64_json": encoded}], "ball", "url", output_format="gif"
                )


class GifProtocolTests(unittest.TestCase):
    def test_generation_protocol_passes_gif_fields(self) -> None:
        with mock.patch.object(openai_v1_image_generations, "stream_image_outputs_with_pool", return_value=[]) as stream:
            openai_v1_image_generations.handle({
                "model": "gpt-image-2",
                "prompt": "ball",
                "output_format": "gif",
                "frame_count": 4,
                "frame_duration_ms": 100,
            })
        request = stream.call_args.args[0]
        self.assertEqual(
            (request.output_format, request.frame_count, request.frame_duration_ms), ("gif", 4, 100)
        )

    def test_edit_rejects_gif(self) -> None:
        with self.assertRaises(ImageGenerationError):
            openai_v1_image_edit.handle({"prompt": "ball", "output_format": "gif", "images": ["x"]})

    def test_pool_rejects_gif_for_image_edits(self) -> None:
        request = ConversationRequest(model="gpt-image-2", output_format="gif", images=["a"])
        with self.assertRaises(ImageGenerationError):
            list(conversation.stream_image_outputs_with_pool(request))

    def test_web_prompt_asks_for_animation(self) -> None:
        prompt = conversation.build_image_prompt("小猫", None, "auto", "gif", 4, 100)
        self.assertIn("GIF", prompt)
        self.assertIn("4 帧", prompt)
        self.assertNotIn("GIF", conversation.build_image_prompt("小猫", None, "auto"))


class CodexGifPayloadTests(unittest.TestCase):
    def _payload(self, **kwargs) -> dict:
        backend = object.__new__(OpenAIBackendAPI)
        backend.access_token = "test-not-a-secret"
        backend.base_url = "https://chatgpt.com"
        with mock.patch.object(backend, "_ensure_codex_source_account"), mock.patch.object(
            backend, "_iter_codex_response_events", return_value=iter([])
        ), mock.patch("services.openai_backend_api.urllib.request.urlopen") as send:
            list(backend.iter_codex_image_response_events("ball", **kwargs))
        import json

        return json.loads(send.call_args.args[0].data)["tools"][0]

    def test_gif_parameters_reach_the_tool(self) -> None:
        tool = self._payload(output_format="gif", frame_count=4, frame_duration_ms=100)
        self.assertEqual(tool["output_format"], "gif")
        self.assertEqual(tool["frame_count"], 4)
        self.assertEqual(tool["frame_duration_ms"], 100)

    def test_png_omits_frame_parameters(self) -> None:
        tool = self._payload()
        self.assertEqual(tool["output_format"], "png")
        self.assertNotIn("frame_count", tool)
        self.assertNotIn("frame_duration_ms", tool)


if __name__ == "__main__":
    unittest.main()
