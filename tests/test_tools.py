from __future__ import annotations

import asyncio
import unittest
from io import BytesIO
from types import SimpleNamespace

import maodou.tools as tools
from maodou.tools import (
    ToolResult,
    document_attachments,
    eval_math,
    extract_document_text,
    fetch_page,
    is_private_host,
    run_shell,
    search_web,
    validate_shell_command,
)


async def _bocha_with_payload(payload: dict) -> ToolResult:
    original = tools.httpx2.AsyncClient
    transport = tools.httpx2.MockTransport(
        lambda request: tools.httpx2.Response(200, json=payload)
    )
    tools.httpx2.AsyncClient = lambda **kwargs: original(transport=transport, **kwargs)
    try:
        return await tools._bocha_search("q", 5, "key", 5)
    finally:
        tools.httpx2.AsyncClient = original


def _make_docx(paragraphs: list[str]) -> bytes:
    from docx import Document

    buffer = BytesIO()
    document = Document()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    document.save(buffer)
    return buffer.getvalue()


class ToolGuardTests(unittest.TestCase):
    def test_document_attachments(self) -> None:
        raw = {
            "attachments": [
                {
                    "content_type": "application/pdf",
                    "filename": "报告.pdf",
                    "url": "https://example.com/report.pdf",
                },
                {
                    "content_type": "application/octet-stream",
                    "filename": "notes.docx",
                    "url": "https://example.com/notes.docx",
                },
                {"content_type": "image/jpeg", "filename": "a.jpg", "url": "https://x/a.jpg"},
                "not-a-dict",
            ]
        }
        attachments = document_attachments(raw)
        self.assertEqual([a.filename for a in attachments], ["报告.pdf", "notes.docx"])

    def test_private_host_detection(self) -> None:
        self.assertTrue(is_private_host("127.0.0.1"))
        self.assertTrue(is_private_host("10.0.0.1"))
        self.assertTrue(is_private_host("192.168.1.1"))
        self.assertTrue(is_private_host("localhost"))
        self.assertFalse(is_private_host("1.1.1.1"))

    def test_eval_math_valid(self) -> None:
        self.assertEqual(eval_math("2 + 3 * 4").output, "14")
        self.assertEqual(eval_math("sqrt(16)").output, "4")
        self.assertEqual(eval_math("round(2 / 3, 2)").output, "0.67")

    def test_eval_math_rejects_unsafe(self) -> None:
        self.assertFalse(eval_math("__import__('os')").ok)
        self.assertFalse(eval_math("[1,2,3]").ok)

    def test_validate_shell_readonly(self) -> None:
        self.assertTrue(validate_shell_command("ls -la", "readonly")[0])
        self.assertTrue(validate_shell_command("echo hello", "readonly")[0])
        self.assertFalse(validate_shell_command("curl http://x", "readonly")[0])
        self.assertFalse(validate_shell_command("python3 -c 'print(1)'", "readonly")[0])
        self.assertFalse(validate_shell_command("echo hi > /tmp/x", "readonly")[0])
        self.assertFalse(validate_shell_command("sudo id", "write")[0])
        self.assertFalse(validate_shell_command("rm -rf /", "write")[0])
        self.assertTrue(validate_shell_command("python3 -c 'print(1)'", "write")[0])


class DocumentParseTests(unittest.TestCase):
    def test_extract_docx_text(self) -> None:
        text = extract_document_text(
            _make_docx(["第一段内容", "第二段内容"]),
            filename="notes.docx",
        )
        self.assertIn("第一段内容", text)
        self.assertIn("第二段内容", text)

    def test_extract_plain_text(self) -> None:
        text = extract_document_text("你好世界".encode("utf-8"), filename="notes.txt")
        self.assertEqual(text, "你好世界")


class ToolExecutionTests(unittest.TestCase):
    def test_search_auto_uses_tavily_when_key_present(self) -> None:
        async def fake_tavily(query, limit, key, timeout):
            return ToolResult(ok=True, output="答案", data=[{"title": "T", "url": "u", "snippet": "s"}])

        original = tools._tavily_search
        tools._tavily_search = fake_tavily
        try:
            result = asyncio.run(
                search_web("q", provider="auto", tavily_api_key="secret")
            )
        finally:
            tools._tavily_search = original
        self.assertTrue(result.ok)
        self.assertEqual(result.output, "答案")

    def test_search_without_any_key_is_not_configured(self) -> None:
        result = asyncio.run(search_web("q", provider="auto"))
        self.assertFalse(result.ok)
        self.assertIn("没有配置", result.error)

    def test_search_never_fakes_success_without_provider(self) -> None:
        # 显式指定提供方但没配 key，同样必须失败，不许出现 ok=True 的静默兜底。
        result = asyncio.run(search_web("q", provider="bocha"))
        self.assertFalse(result.ok)
        self.assertIn("key", result.error)

    def test_search_rejects_unknown_provider(self) -> None:
        result = asyncio.run(search_web("q", provider="bing_web", bocha_api_key="k"))
        self.assertFalse(result.ok)
        self.assertIn("不支持", result.error)

    def test_search_auto_prefers_bocha(self) -> None:
        seen: list[str] = []

        async def fake_bocha(query, limit, key, timeout):
            seen.append(query)
            return ToolResult(ok=True, data=[{"title": "B", "url": "u", "snippet": "s"}])

        original = tools._bocha_search
        tools._bocha_search = fake_bocha
        try:
            result = asyncio.run(
                search_web("q", provider="auto", bocha_api_key="k", tavily_api_key="t")
            )
        finally:
            tools._bocha_search = original
        self.assertTrue(result.ok)
        self.assertEqual(seen, ["q"])

    def test_search_falls_back_from_bocha_to_tavily(self) -> None:
        async def fake_bocha(query, limit, key, timeout):
            return ToolResult(ok=False, error="bocha down")

        async def fake_tavily(query, limit, key, timeout):
            return ToolResult(ok=True, data=[{"title": "T", "url": "u", "snippet": "s"}])

        original_bocha = tools._bocha_search
        original_tavily = tools._tavily_search
        tools._bocha_search = fake_bocha
        tools._tavily_search = fake_tavily
        try:
            result = asyncio.run(
                search_web("q", provider="auto", bocha_api_key="k", tavily_api_key="t")
            )
        finally:
            tools._bocha_search = original_bocha
            tools._tavily_search = original_tavily
        self.assertTrue(result.ok)
        self.assertEqual(result.data[0]["title"], "T")

    def test_search_reports_error_when_every_provider_fails(self) -> None:
        async def fake_bocha(query, limit, key, timeout):
            return ToolResult(ok=False, error="boom")

        original = tools._bocha_search
        tools._bocha_search = fake_bocha
        try:
            result = asyncio.run(search_web("q", provider="auto", bocha_api_key="k"))
        finally:
            tools._bocha_search = original
        self.assertFalse(result.ok)
        self.assertIn("boom", result.error)

    def test_bocha_search_parses_results(self) -> None:
        payload = {
            "code": 200,
            "data": {
                "webPages": {
                    "value": [
                        {"name": "标题A", "url": "https://x/a", "summary": "摘要A"},
                        {"name": "标题B", "url": "https://x/b", "snippet": "摘要B"},
                    ]
                }
            },
        }
        result = asyncio.run(_bocha_with_payload(payload))
        self.assertTrue(result.ok)
        self.assertEqual(len(result.data), 2)
        self.assertEqual(result.data[0]["title"], "标题A")
        self.assertEqual(result.data[1]["snippet"], "摘要B")

    def test_bocha_search_empty_result_is_not_ok(self) -> None:
        payload = {"code": 200, "data": {"webPages": {"value": []}}}
        result = asyncio.run(_bocha_with_payload(payload))
        self.assertFalse(result.ok)
        self.assertIn("没有返回", result.error)

    def test_fetch_page_rejects_private_host(self) -> None:
        result = asyncio.run(fetch_page("http://127.0.0.1/anything"))
        self.assertFalse(result.ok)

    def test_run_shell_readonly_blocks_without_subprocess(self) -> None:
        result = asyncio.run(run_shell("curl http://x", cwd=".", mode="readonly"))
        self.assertFalse(result.ok)

    def test_run_shell_executes_allowlisted_command(self) -> None:
        original = tools._run_subprocess
        tools._run_subprocess = lambda command, cwd, timeout: SimpleNamespace(
            returncode=0, stdout="hello\n"
        )
        try:
            result = asyncio.run(run_shell("echo hello", cwd=".", mode="readonly"))
        finally:
            tools._run_subprocess = original
        self.assertTrue(result.ok)
        self.assertIn("hello", result.output)


if __name__ == "__main__":
    unittest.main()
