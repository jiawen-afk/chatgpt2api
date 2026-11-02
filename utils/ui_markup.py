from __future__ import annotations

import re

# ChatGPT web 的富文本组件标记（GenUI 风格 DSL）。网页端会把这些标签渲染成卡片，
# API 客户端只认 Markdown，原样透传就变成一堆 <box>/<row> 噪音，所以这里做降级转换。

BLOCK_TAGS = {
    "box",
    "row",
    "column",
    "col",
    "stack",
    "card",
    "grid",
    "group",
    "section",
    "form",
    "radio-group",
    "radio",
    "checkbox",
    "switch",
    "select",
    "slider",
    "input",
    "button",
    "link",
    "chip",
}

BOLD_TAGS = {"title", "badge"}

INLINE_TAGS = {"text", "span", "strong", "em", "label", "p", "small"}

VOID_TAGS = {"asyncimage", "img", "image", "entity", "divider", "icon", "br", "star"}

COMPONENT_TAGS = BLOCK_TAGS | BOLD_TAGS | INLINE_TAGS | VOID_TAGS

_TAG_RE = re.compile(
    r"<(?P<closing>/)?(?P<name>[A-Za-z][\w-]*)"
    r"(?P<attrs>(?:\s+(?:[^>\"'{]|\"[^\"]*\"|'[^']*'|\{(?:[^{}]|\{[^{}]*\})*\})*)?)"
    r"(?P<selfclose>/)?>"
)

_TAG_NAME_RE = re.compile(r"</?([A-Za-z][\w-]*)")

_ATTR_RE = re.compile(r"([A-Za-z_][\w-]*)\s*=\s*(?:\"([^\"]*)\"|'([^']*)')")

MAX_PENDING_TAG_LENGTH = 500


def parse_attributes(raw: str) -> dict[str, str]:
    attributes: dict[str, str] = {}
    for match in _ATTR_RE.finditer(raw or ""):
        attributes[match.group(1).lower()] = match.group(2) if match.group(2) is not None else match.group(3)
    return attributes


def convert_void_tag(name: str, attributes: dict[str, str]) -> str:
    if name == "entity":
        return str(attributes.get("value") or attributes.get("name") or "").strip()
    if name == "divider":
        return "\n\n---\n\n"
    if name == "br":
        return "\n"
    if name in {"img", "image"}:
        source = str(attributes.get("src") or "").strip()
        alt = str(attributes.get("alt") or "").strip()
        if source:
            return f"![{alt}]({source})"
    # AsyncImage 只有一句用于检索配图的 query，没有真实地址，直接丢弃。
    return ""


def is_component_tag_name(text: str, index: int) -> bool:
    """判断 `<` 后面跟的是不是组件名（含被写残的前缀，例如 `<Async`）。"""
    match = _TAG_NAME_RE.match(text, index)
    if not match:
        return False
    name = match.group(1).lower()
    if name in COMPONENT_TAGS:
        return True
    return len(name) >= 3 and any(tag.startswith(name) for tag in COMPONENT_TAGS)


def collapse_blank_lines(text: str) -> str:
    lines = text.split("\n")
    output: list[str] = []
    in_fence = False
    blank = False
    for line in lines:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            output.append(line)
            blank = False
            continue
        if in_fence:
            output.append(line)
            continue
        if not line.strip():
            if blank:
                continue
            blank = True
            output.append("")
            continue
        blank = False
        output.append(line.rstrip())
    return "\n".join(output)


def strip_ui_markup(text: str) -> str:
    """把上游返回的组件标记降级成 Markdown，代码块与行内代码原样保留。"""
    text = str(text or "")
    if "<" not in text and "{@" not in text:
        return text

    output: list[str] = []
    index = 0
    length = len(text)
    in_fence = False
    depth = 0
    at_line_start = True

    def emit(value: str) -> None:
        nonlocal at_line_start
        if not value:
            return
        output.append(value)
        at_line_start = value.endswith("\n")

    while index < length:
        if text.startswith("```", index):
            in_fence = not in_fence
            emit("```")
            index += 3
            continue
        if in_fence:
            emit(text[index])
            index += 1
            continue
        if text[index] == "`":
            end = text.find("`", index + 1)
            if end == -1:
                emit(text[index:])
                break
            emit(text[index:end + 1])
            index = end + 1
            continue
        if text.startswith("{@", index):
            # {@body const [x,setX] = DIL.useState(...)} 这类脚本行，整行丢弃。
            line_end = text.find("\n", index)
            index = length if line_end == -1 else line_end
            at_line_start = False
            continue
        if text[index] == "\n":
            emit("\n")
            index += 1
            continue
        if text[index] in " \t" and at_line_start and depth > 0:
            # 组件内部的缩进是布局用的，去掉标签后会变成 Markdown 代码块。
            index += 1
            continue
        if text[index] == "<":
            match = _TAG_RE.match(text, index)
            if match:
                name = match.group("name").lower()
                if name in COMPONENT_TAGS:
                    closing = bool(match.group("closing"))
                    if name in BLOCK_TAGS:
                        if closing:
                            depth = max(0, depth - 1)
                        elif not match.group("selfclose"):
                            depth += 1
                    elif name in VOID_TAGS or match.group("selfclose"):
                        emit(convert_void_tag(name, parse_attributes(match.group("attrs"))))
                    elif name in BOLD_TAGS:
                        emit("**")
                    index = match.end()
                    continue
            if is_component_tag_name(text, index):
                line_end = text.find("\n", index)
                if 0 <= line_end - index <= MAX_PENDING_TAG_LENGTH:
                    # 组件名后面没闭合就换行了：这是上游写残的标签，整行丢掉，保留后面的正文。
                    index = line_end + 1
                    continue
                if len(text) - index <= MAX_PENDING_TAG_LENGTH:
                    # 还没换行也没闭合，可能是流式切开，先扣住不发。
                    break
            emit("<")
            index += 1
            continue
        emit(text[index])
        index += 1

    return collapse_blank_lines("".join(output)).strip()
