from __future__ import annotations

import json
import unittest
from contextlib import contextmanager

from services.config import config
from services.protocol.conversation import (
    conversation_events,
    iter_conversation_payloads,
    normalize_messages,
)


@contextmanager
def variant_policy(value: str):
    original = config.data.get("stream_variant_policy")
    config.data["stream_variant_policy"] = value
    try:
        yield
    finally:
        if original is None:
            config.data.pop("stream_variant_policy", None)
        else:
            config.data["stream_variant_policy"] = original


def append_event(text: str) -> str:
    return json.dumps({"p": "/message/content/parts/0", "o": "append", "v": text})


def message_event(text: str, message_id: str) -> str:
    return json.dumps({
        "o": "add",
        "v": {
            "message": {
                "id": message_id,
                "author": {"role": "assistant"},
                "content": {"content_type": "text", "parts": [text]},
            }
        },
    })


def collect_deltas(payloads: list[str], history_text: str = "", history_messages: list[str] | None = None) -> str:
    parts = []
    for event in iter_conversation_payloads(iter(payloads), history_text, history_messages):
        if event.get("type") == "conversation.delta":
            parts.append(str(event.get("delta") or ""))
    return "".join(parts)


class StreamTextDedupTests(unittest.TestCase):
    def test_new_message_snapshot_is_not_replayed_wholesale(self) -> None:
        answer = "少爷今天有什么我可以效劳的吗？\n\n关于这次升级，我的建议是先看显存。"
        payloads = [append_event("少爷今天有什么我可以效劳的吗？"), append_event("\n\n关于这次升级，")]
        # Upstream restarts the answer as a second message: every snapshot grows from the
        # greeting again, which used to be emitted in full on every single event.
        for end in range(1, len(answer) + 1):
            payloads.append(message_event(answer[:end], "msg-new"))
        payloads.append("[DONE]")

        with variant_policy("merge"):
            self.assertEqual(collect_deltas(payloads), answer)

    def test_parallel_variant_keeps_one_copy(self) -> None:
        # 上游 variant_purpose=comparison_implicit 会并行生成两条高度相似的消息，
        # 它们交替推送快照；落后那条不重复发送。
        variant_a = "少爷，A 方案更省心，适合长期运行。"
        variant_b = "少爷，A 方案更省心，适合长期运行，B 方案更快。"
        payloads = [
            message_event(variant_a[:8], "msg-a"),
            message_event(variant_b[:8], "msg-b"),
            message_event(variant_a[:16], "msg-a"),
            message_event(variant_b[:16], "msg-b"),
            message_event(variant_a, "msg-a"),
            message_event(variant_b, "msg-b"),
            "[DONE]",
        ]

        self.assertEqual(collect_deltas(payloads), variant_a)

    def test_shrinking_snapshot_does_not_duplicate_streamed_text(self) -> None:
        payloads = [
            append_event("Hello "),
            append_event(","),
            append_event(" world"),
            "[DONE]",
        ]

        output = collect_deltas(payloads)
        self.assertEqual(output.count("Hello"), 1)
        self.assertIn("world", output)

    def test_section_retake_is_stitched_onto_the_emitted_tail(self) -> None:
        first = "少爷，建议先看显存。\n\n## 二、实际生图能力差多少？\n\n少爷，我查到一份 AMD 发布的"
        retake = "## 二、实际生图能力差多少？\n\n少爷，我查到一份 AMD 发布的官方测试，比较适合判断能做什么。"
        payloads = [message_event(first, "msg-1"), message_event(retake, "msg-2"), "[DONE]"]

        with variant_policy("merge"):
            output = collect_deltas(payloads)
        self.assertEqual(output, first + "官方测试，比较适合判断能做什么。")
        self.assertEqual(output.count("## 二、"), 1)

    def test_section_retake_with_leading_blank_lines_is_not_repeated(self) -> None:
        header = "## 二、实际生图能力差多少？"
        para = "截至 2026 年 10 月，AMD 已经把本地生图的支持做得比早期完善不少。"
        payloads = [
            message_event("这条路线也许比换整台电脑更划算。", "msg-1"),
            # 第一次重来贴着上一句，后续快照都带前导空行。
            message_event(header, "msg-2"),
        ]
        for end in (8, 20, len(para)):
            payloads.append(message_event("\n\n" + header + "\n\n" + para[:end], "msg-2"))
        payloads.append("[DONE]")

        with variant_policy("merge"):
            output = collect_deltas(payloads)
        self.assertEqual(output.count(header), 1)
        self.assertEqual(output.count("截至 2026 年 10 月"), 1)
        self.assertTrue(output.endswith(para))

    def test_second_message_id_is_dropped_in_follow_first_mode(self) -> None:
        first = "少爷，这是第一条消息的完整回答。"
        variant = "少爷，这是另一份措辞不同的回答。"
        payloads = [message_event(first, "msg-1"), message_event(variant, "msg-2"), "[DONE]"]

        with variant_policy("follow_first"):
            self.assertEqual(collect_deltas(payloads), first)

    def test_stalled_message_does_not_truncate_the_answer(self) -> None:
        # 线上日志：A 停在 299 字，B 一路长到 1632 字。extend 策略下答案不能被截断。
        common = "少爷，先说结论：这条路线更划算。" + "显存和内存带宽决定了能跑多大的模型。" * 12
        answer_a = common + "A 的后续内容。" * 10
        answer_b = common + "B 的后续内容。" * 40

        payloads = [
            message_event(answer_a[:25], "msg-a"),
            message_event(answer_b[:25], "msg-b"),
            message_event(answer_b[:222], "msg-b"),
            message_event(answer_a[:280], "msg-a"),
            message_event(answer_b[:280], "msg-b"),
            message_event(answer_a, "msg-a"),
            message_event(answer_b[:299], "msg-b"),
            message_event("标题", "msg-c"),
            message_event(answer_b, "msg-b"),
            "[DONE]",
        ]

        output = collect_deltas(payloads)
        self.assertEqual(output.count("少爷，先说结论"), 1)
        self.assertEqual(len(output), max(len(answer_a), len(answer_b)))
        self.assertTrue(output.endswith(answer_b[-12:]))

    def test_stream_stops_when_upstream_marks_the_message_finished(self) -> None:
        # 上游写完一条消息后会长时间保持连接，只等 [DONE] 会让客户端在答案完整后干等。
        finished = json.dumps({"o": "add", "v": {"message": {
            "id": "m1",
            "author": {"role": "assistant"},
            "status": "finished_successfully",
            "content": {"content_type": "text", "parts": ["第一段。"]},
        }}})
        payloads = [message_event("第一段。", "m1"), finished, message_event("不该被读到", "m1")]

        events = list(iter_conversation_payloads(iter(payloads), stop_on_finish=True))
        text = "".join(str(event.get("delta") or "") for event in events if event.get("type") == "conversation.delta")
        self.assertEqual(text, "第一段。")
        self.assertEqual(sum(1 for event in events if event.get("type") == "conversation.done"), 1)

    def test_history_replay_is_skipped(self) -> None:
        previous = "上一轮的回答。"
        payloads = [
            message_event(previous, "msg-old"),
            append_event("本轮新回答。"),
            "[DONE]",
        ]

        output = collect_deltas(payloads, history_text=previous, history_messages=[previous])
        self.assertEqual(output, "本轮新回答。")

    def test_global_system_prompt_is_injected_once(self) -> None:
        original = config.data.get("global_system_prompt")
        config.data["global_system_prompt"] = "你是严格的代码审查员。"
        try:
            messages = normalize_messages([{"role": "user", "content": "介绍下你自己"}])
            normalized_again = normalize_messages(messages)
        finally:
            if original is None:
                config.data.pop("global_system_prompt", None)
            else:
                config.data["global_system_prompt"] = original

        self.assertEqual(sum(1 for item in messages if item["role"] == "system"), 1)
        self.assertEqual(sum(1 for item in normalized_again if item["role"] == "system"), 1)


class ConversationEventsNormalizationTests(unittest.TestCase):
    def test_pre_normalized_messages_are_not_reinjected(self) -> None:
        original = config.data.get("global_system_prompt")
        config.data["global_system_prompt"] = "全局提示。"
        captured: dict[str, object] = {}

        class FakeBackend:
            access_token = "token-1"

            def stream_conversation(self, messages=None, **_kwargs):
                captured["messages"] = messages
                yield "[DONE]"

            def close(self):
                pass

        try:
            messages = normalize_messages([{"role": "user", "content": "在吗"}])
            list(conversation_events(FakeBackend(), messages=messages, model="auto"))
        finally:
            if original is None:
                config.data.pop("global_system_prompt", None)
            else:
                config.data["global_system_prompt"] = original

        sent = captured["messages"]
        self.assertEqual(sum(1 for item in sent if item["role"] == "system"), 1)


if __name__ == "__main__":
    unittest.main()
