from __future__ import annotations

import unittest

from services.config import config
from services.protocol.conversation import sanitize_output_text
from utils.ui_markup import strip_ui_markup

SAMPLE = (
    "<box gap={3}>\n"
    '  <row align=start gap={3}>\n'
    '    <AsyncImage query="AMD Ryzen AI Max Strix Halo mini PC Radeon 8060S workstation" aspectRatio="4:5" maxWidth="124px"/>\n'
    '    <box flex="1" gap={1}>\n'
    '      <title size="lg">路线 A：AMD AI Max 系列</title>\n'
    '      <badge color="success">优先考虑</badge>\n'
    "      **适合：本地生图、模型实验、多模型共存**\n"
    "    </box>\n"
    "  </row>\n"
    "</box>\n"
    "\n"
    "这组数据说明……\n"
)


class UiMarkupTests(unittest.TestCase):
    def test_component_block_becomes_markdown(self) -> None:
        self.assertEqual(
            strip_ui_markup(SAMPLE),
            "**路线 A：AMD AI Max 系列**\n"
            "**优先考虑**\n"
            "**适合：本地生图、模型实验、多模型共存**\n"
            "\n"
            "这组数据说明……",
        )

    def test_component_indent_does_not_become_code_block(self) -> None:
        output = strip_ui_markup("<box>\n      - 体积小、功耗低\n      - 适合长期运行\n</box>")
        self.assertEqual(output, "- 体积小、功耗低\n- 适合长期运行")

    def test_void_tags(self) -> None:
        self.assertEqual(strip_ui_markup('<Entity value="Apple Mac mini" category="product"/>'), "Apple Mac mini")
        self.assertEqual(strip_ui_markup("上文\n<divider color=\"subtle\"/>\n下文"), "上文\n\n---\n\n下文")
        self.assertEqual(strip_ui_markup('<AsyncImage query="a mini pc" aspectRatio="4:3"/>'), "")

    def test_script_lines_are_dropped(self) -> None:
        source = '{@body const [budget,setBudget] = DIL.useState("8000-12000")}\n正文'
        self.assertEqual(strip_ui_markup(source), "正文")

    def test_code_fence_is_untouched(self) -> None:
        source = "示例：\n```html\n<box gap={3}>\n  <title>标题</title>\n</box>\n```\n完成"
        self.assertEqual(strip_ui_markup(source), source)

    def test_inline_code_is_untouched(self) -> None:
        source = "参数写成 `<box gap={3}>` 这样。"
        self.assertEqual(strip_ui_markup(source), source)

    def test_incomplete_tag_is_held_back(self) -> None:
        self.assertEqual(strip_ui_markup("前文 <box gap={3"), "前文")
        self.assertEqual(strip_ui_markup("前文 <AsyncImage query=\"mini"), "前文")

    def test_broken_tag_on_its_own_line_keeps_the_following_content(self) -> None:
        # 线上实际出现过：上游只写出 `<Async` 就换行，后面紧跟表格。
        source = "目标总价：约 6,500 元\n<Async\n| 建议规格 |\n|---|---|\n| 显卡 | RTX 5060 Ti |"
        self.assertEqual(
            strip_ui_markup(source),
            "目标总价：约 6,500 元\n| 建议规格 |\n|---|---|\n| 显卡 | RTX 5060 Ti |",
        )

    def test_arrow_function_attribute_does_not_break_the_tag(self) -> None:
        source = '<button block onClick={() => GenUI.issueNewTurn("帮我选配置")}>帮我选具体配置</button>'
        self.assertEqual(strip_ui_markup(source), "帮我选具体配置")

    def test_nested_braces_in_attribute(self) -> None:
        source = (
            '<button block onClick={() => GenUI.issueNewTurn("预算："+budget'
            '+({"sdxl":"静态生图","flux":"Flux"}[workflow]))}>提交配置</button>'
        )
        self.assertEqual(strip_ui_markup(source), "提交配置")

    def test_plain_text_is_untouched(self) -> None:
        source = "普通文本，包含 a < b 这种比较符号。"
        self.assertEqual(strip_ui_markup(source), source)

    def test_sanitize_output_text_applies_stripping(self) -> None:
        original = config.data.get("strip_ui_markup")
        config.data["strip_ui_markup"] = True
        try:
            self.assertEqual(sanitize_output_text("<box>\n  <title>标题</title>\n</box>"), "**标题**")
        finally:
            if original is None:
                config.data.pop("strip_ui_markup", None)
            else:
                config.data["strip_ui_markup"] = original

    def test_stripping_can_be_disabled(self) -> None:
        original = config.data.get("strip_ui_markup")
        config.data["strip_ui_markup"] = False
        try:
            source = "<box gap={3}>内容</box>"
            self.assertEqual(sanitize_output_text(source), source)
        finally:
            if original is None:
                config.data.pop("strip_ui_markup", None)
            else:
                config.data["strip_ui_markup"] = original


if __name__ == "__main__":
    unittest.main()
