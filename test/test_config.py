import json
import tempfile
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
ROOT_CONFIG_FILE = ROOT_DIR / "config.json"


class ConfigLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._created_root_config = False
        if not ROOT_CONFIG_FILE.exists():
            ROOT_CONFIG_FILE.write_text(json.dumps({"auth-key": "test-auth"}), encoding="utf-8")
            cls._created_root_config = True

        from services import config as config_module

        cls.config_module = config_module

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._created_root_config and ROOT_CONFIG_FILE.exists():
            ROOT_CONFIG_FILE.unlink()

    def test_load_settings_ignores_directory_config_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            data_dir = base_dir / "data"
            config_dir = base_dir / "config.json"
            os_auth_key = "env-auth"

            config_dir.mkdir()

            module = self.config_module
            old_base_dir = module.BASE_DIR
            old_data_dir = module.DATA_DIR
            old_config_file = module.CONFIG_FILE
            old_env_auth_key = module.os.environ.get("CHATGPT2API_AUTH_KEY")
            try:
                module.BASE_DIR = base_dir
                module.DATA_DIR = data_dir
                module.CONFIG_FILE = config_dir
                module.os.environ["CHATGPT2API_AUTH_KEY"] = os_auth_key

                settings = module._load_settings()

                self.assertEqual(settings.auth_key, os_auth_key)
                self.assertEqual(settings.refresh_account_interval_minute, 5)
            finally:
                module.BASE_DIR = old_base_dir
                module.DATA_DIR = old_data_dir
                module.CONFIG_FILE = old_config_file
                if old_env_auth_key is None:
                    module.os.environ.pop("CHATGPT2API_AUTH_KEY", None)
                else:
                    module.os.environ["CHATGPT2API_AUTH_KEY"] = old_env_auth_key

    def test_image_timeout_retry_secs_is_normalized_in_public_config(self) -> None:
        module = self.config_module
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_file = Path(tmp_dir) / "config.json"
            config_file.write_text(
                json.dumps({"auth-key": "test-auth", "image_timeout_retry_secs": "0"}),
                encoding="utf-8",
            )
            store = module.ConfigStore(config_file)

            self.assertEqual(store.image_timeout_retry_secs, 1)
            self.assertEqual(store.get()["image_timeout_retry_secs"], 1)

    def test_image_models_include_builtin_versions_and_normalize_custom_versions(self) -> None:
        module = self.config_module
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_file = Path(tmp_dir) / "config.json"
            config_file.write_text(
                json.dumps({
                    "auth-key": "test-auth",
                    "custom_image_models": [
                        " GPT-5-6 ",
                        "gpt-5-5",
                        "",
                        "custom-image-v1",
                        "gpt-5-6",
                    ],
                }),
                encoding="utf-8",
            )
            store = module.ConfigStore(config_file)

            self.assertEqual(store.custom_image_models, ["gpt-5-6", "custom-image-v1"])
            self.assertEqual(
                store.image_models,
                [
                    "gpt-image-2.6-sunburst",
                    "gpt-image-2.6-flare",
                    "gpt-image-2.5-sunburst",
                    "gpt-image-2.5-flare",
                    "gpt-image-2",
                    "gpt-5-5-thinking",
                    "gpt-5-5",
                    "gpt-5-3",
                    "gpt-5-6",
                    "custom-image-v1",
                ],
            )
            self.assertEqual(store.get()["custom_image_models"], ["gpt-5-6", "custom-image-v1"])
            self.assertEqual(
                store.get()["image_models"],
                [
                    "gpt-image-2.6-sunburst",
                    "gpt-image-2.6-flare",
                    "gpt-image-2.5-sunburst",
                    "gpt-image-2.5-flare",
                    "gpt-image-2",
                    "gpt-5-5-thinking",
                    "gpt-5-5",
                    "gpt-5-3",
                    "gpt-5-6",
                    "custom-image-v1",
                ],
            )

    def test_update_persists_refreshed_image_model_cache(self) -> None:
        module = self.config_module
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_file = Path(tmp_dir) / "config.json"
            config_file.write_text(json.dumps({"auth-key": "test-auth"}), encoding="utf-8")
            store = module.ConfigStore(config_file)

            saved = store.update({
                "image_models_cache": ["gpt-image-2.5-sunburst", "gpt-image-2", "gpt-5-7"],
                "image_models_source": "upstream",
                "image_models_updated_at": "2026-09-20T12:00:00+00:00",
            })

            self.assertEqual(saved["image_models"], [*module.DEFAULT_IMAGE_MODELS, "gpt-5-7"])
            self.assertEqual(saved["image_models_source"], "upstream")
            self.assertEqual(saved["image_models_updated_at"], "2026-09-20T12:00:00+00:00")

    def test_new_builtin_models_can_be_selected_with_legacy_cache(self) -> None:
        module = self.config_module
        for model in ("gpt-image-2.6-flare", "gpt-image-2.6-sunburst"):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as tmp_dir:
                config_file = Path(tmp_dir) / "config.json"
                config_file.write_text(json.dumps({
                    "auth-key": "test-auth",
                    "image_models_cache": ["gpt-image-2.5-sunburst", "gpt-image-2", "gpt-5-7"],
                    "default_image_model": "gpt-image-2.5-sunburst",
                }), encoding="utf-8")
                store = module.ConfigStore(config_file)

                self.assertIn(model, store.image_models)
                self.assertIn("gpt-5-7", store.image_models)
                self.assertEqual(store.default_image_model, "gpt-image-2.5-sunburst")
                self.assertEqual(len(store.image_models), len(set(store.image_models)))

                saved = store.update({"default_image_model": model})
                reloaded = module.ConfigStore(config_file)

                self.assertEqual(saved["default_image_model"], model)
                self.assertEqual(reloaded.default_image_model, model)
                self.assertIn(model, reloaded.get()["image_models"])

    def test_default_image_model_is_saved_and_reloaded(self) -> None:
        module = self.config_module
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_file = Path(tmp_dir) / "config.json"
            config_file.write_text(json.dumps({"auth-key": "test-auth"}), encoding="utf-8")
            store = module.ConfigStore(config_file)

            saved = store.update({
                "custom_image_models": ["gpt-image-2.5-sunburst"],
                "default_image_model": "gpt-image-2.5-sunburst",
            })
            reloaded = module.ConfigStore(config_file)

            self.assertEqual(saved["default_image_model"], "gpt-image-2.5-sunburst")
            self.assertEqual(reloaded.default_image_model, "gpt-image-2.5-sunburst")
            self.assertEqual(reloaded.get()["default_image_model"], "gpt-image-2.5-sunburst")

    def test_default_image_model_falls_back_when_custom_model_is_removed(self) -> None:
        module = self.config_module
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_file = Path(tmp_dir) / "config.json"
            config_file.write_text(
                json.dumps({
                    "auth-key": "test-auth",
                    "custom_image_models": ["custom-image-v1"],
                    "default_image_model": "custom-image-v1",
                }),
                encoding="utf-8",
            )
            store = module.ConfigStore(config_file)

            saved = store.update({"custom_image_models": []})

            self.assertEqual(saved["default_image_model"], "gpt-image-2")


if __name__ == "__main__":
    unittest.main()
