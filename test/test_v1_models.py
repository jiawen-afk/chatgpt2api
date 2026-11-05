from __future__ import annotations

import json
import unittest
from unittest import mock

import requests

from services.protocol import openai_v1_models


AUTH_KEY = "chatgpt2api"
BASE_URL = "http://localhost:8000"


class ModelListTests(unittest.TestCase):
    def test_list_models_only_returns_image_models_backed_by_account_types(self):
        with (
            mock.patch.object(
                openai_v1_models.OpenAIBackendAPI,
                "list_models",
                return_value={"object": "list", "data": []},
            ),
            mock.patch.object(
                openai_v1_models.account_service,
                "list_accounts",
                return_value=[
                    {"access_token": "token-free", "type": "free"},
                    {"access_token": "token-web-team", "type": "Team", "source_type": "web"},
                    {"access_token": "token-codex-team", "type": "Team", "source_type": "codex"},
                ],
            ),
        ):
            result = openai_v1_models.list_models()

        ids = {item["id"] for item in result["data"]}
        self.assertIn("gpt-image-2", ids)
        self.assertIn("gpt-image-2.6-flare", ids)
        self.assertIn("gpt-image-2.6-sunburst", ids)
        self.assertIn("codex-gpt-image-2", ids)
        self.assertIn("team-codex-gpt-image-2", ids)
        self.assertNotIn("plus-codex-gpt-image-2", ids)
        self.assertNotIn("pro-codex-gpt-image-2", ids)

    def test_list_models_does_not_return_codex_models_for_web_plus_accounts(self):
        with (
            mock.patch.object(
                openai_v1_models.OpenAIBackendAPI,
                "list_models",
                return_value={"object": "list", "data": []},
            ),
            mock.patch.object(
                openai_v1_models.account_service,
                "list_accounts",
                return_value=[
                    {"access_token": "token-web-plus", "type": "Plus", "source_type": "web"},
                ],
            ),
        ):
            result = openai_v1_models.list_models()

        ids = {item["id"] for item in result["data"]}
        self.assertIn("gpt-image-2", ids)
        self.assertNotIn("codex-gpt-image-2", ids)
        self.assertNotIn("plus-codex-gpt-image-2", ids)

    def test_list_models_exposes_configured_web_image_versions(self):
        original_custom_models = openai_v1_models.config.data.get("custom_image_models")
        original_default_model = openai_v1_models.config.data.get("default_image_model")
        openai_v1_models.config.data["custom_image_models"] = ["custom-image-v1"]
        openai_v1_models.config.data["default_image_model"] = "custom-image-v1"
        try:
            with (
                mock.patch.object(
                    openai_v1_models.OpenAIBackendAPI,
                    "list_models",
                    return_value={"object": "list", "data": [
                        {
                            "id": "gpt-5-5",
                            "object": "model",
                            "created": 0,
                            "owned_by": "chatgpt",
                            "permission": [],
                            "root": "gpt-5-5",
                            "parent": None,
                        }
                    ]},
                ),
                mock.patch.object(
                    openai_v1_models.account_service,
                    "list_accounts",
                    return_value=[
                        {"access_token": "token-web-plus", "type": "Plus", "source_type": "web"},
                    ],
                ),
            ):
                result = openai_v1_models.list_models()
        finally:
            if original_custom_models is None:
                openai_v1_models.config.data.pop("custom_image_models", None)
            else:
                openai_v1_models.config.data["custom_image_models"] = original_custom_models
            if original_default_model is None:
                openai_v1_models.config.data.pop("default_image_model", None)
            else:
                openai_v1_models.config.data["default_image_model"] = original_default_model

        models_by_id = {item["id"]: item for item in result["data"]}
        self.assertIn("gpt-image-2", models_by_id)
        self.assertIn("gpt-5-5-thinking", models_by_id)
        self.assertIn("gpt-5-5", models_by_id)
        self.assertIn("gpt-5-3", models_by_id)
        self.assertIn("custom-image-v1", models_by_id)
        self.assertEqual(models_by_id["gpt-5-5"]["owned_by"], "chatgpt2api")
        self.assertEqual(result["default_image_model"], "custom-image-v1")

    def test_list_models_function(self):
        """测试直接调用服务层获取模型列表。"""
        result = openai_v1_models.list_models()
        print("function result:")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    def test_list_models_http(self):
        """测试通过 HTTP 接口获取模型列表。"""
        response = requests.get(
            f"{BASE_URL}/v1/models",
            headers={"Authorization": f"Bearer {AUTH_KEY}"},
            timeout=30,
        )
        print("http status:")
        print(response.status_code)
        print("http result:")
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
