from __future__ import annotations

import asyncio
import json
import unittest
from io import BytesIO
from types import SimpleNamespace

import maodou.tools as tools
from maodou.bot import MaodouClient
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
    def _use_transport(self, handler):
        original = tools.httpx2.AsyncClient
        transport = tools.httpx2.MockTransport(handler)
        tools.httpx2.AsyncClient = lambda **kwargs: original(
            transport=transport, **kwargs
        )
        return original

    def test_search_without_key_is_not_configured(self) -> None:
        result = asyncio.run(search_web("q"))
        self.assertFalse(result.ok)
        self.assertIn("没有配置", result.error)
        self.assertIn("LLM_API_KEY", result.error)

    def test_search_parses_deepseek_results_and_sends_expected_request(self) -> None:
        captured: list[tuple[str, str, dict]] = []

        def handler(request):
            captured.append(
                (
                    str(request.url),
                    request.headers.get("authorization", ""),
                    json.loads(request.content),
                )
            )
            return tools.httpx2.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "腹泻常见原因",
                            "url": "https://example.com/a",
                            "summary": "摘要A",
                            "site_name": "example.com",
                            "date": "2026-10-20",
                            "authority": 4,
                        },
                        {
                            "title": "腹痛就医提示",
                            "url": "https://example.com/b",
                            "snippet": "摘要B",
                        },
                    ]
                },
            )

        original = self._use_transport(handler)
        try:
            result = asyncio.run(
                search_web(
                    "腹泻 腹痛 原因",
                    limit=8,
                    api_key="secret",
                    base_url="https://api.deepseek.com",
                    original_question="帮我查查肚子疼拉肚子有没有医学依据",
                )
            )
        finally:
            tools.httpx2.AsyncClient = original

        self.assertTrue(result.ok)
        self.assertEqual(len(result.data), 2)
        self.assertEqual(result.data[0]["snippet"], "摘要A")
        self.assertEqual(result.data[1]["snippet"], "摘要B")
        self.assertEqual(captured[0][0], "https://api.deepseek.com/v1/tools/search")
        self.assertEqual(captured[0][1], "Bearer secret")
        self.assertEqual(
            captured[0][2],
            {
                "query": "腹泻 腹痛 原因",
                "count": 8,
                "need_content": False,
                "ori_question": "帮我查查肚子疼拉肚子有没有医学依据",
            },
        )

    def test_search_truncates_original_question_to_128_chars(self) -> None:
        captured: list[dict] = []

        def handler(request):
            captured.append(json.loads(request.content))
            return tools.httpx2.Response(
                200,
                json={
                    "results": [
                        {"title": "T", "url": "https://example.com", "snippet": "S"}
                    ]
                },
            )

        original = self._use_transport(handler)
        try:
            asyncio.run(search_web("q", api_key="k", original_question="问" * 300))
        finally:
            tools.httpx2.AsyncClient = original

        self.assertEqual(len(captured[0]["ori_question"]), 128)

    def test_search_empty_results_is_not_ok(self) -> None:
        original = self._use_transport(
            lambda request: tools.httpx2.Response(200, json={"results": []})
        )
        try:
            result = asyncio.run(search_web("q", api_key="k"))
        finally:
            tools.httpx2.AsyncClient = original
        self.assertFalse(result.ok)
        self.assertIn("没有返回", result.error)

    def test_search_reports_insufficient_balance_without_retry(self) -> None:
        calls: list[int] = []

        def handler(request):
            calls.append(1)
            return tools.httpx2.Response(
                402, json={"error": {"message": "Insufficient Balance"}}
            )

        original = self._use_transport(handler)
        try:
            result = asyncio.run(search_web("q", api_key="k"))
        finally:
            tools.httpx2.AsyncClient = original
        self.assertFalse(result.ok)
        self.assertIn("余额不足", result.error)
        self.assertEqual(len(calls), 1)

    def test_search_retries_once_after_503(self) -> None:
        statuses = iter([503, 200])

        def handler(request):
            status = next(statuses)
            if status == 503:
                return tools.httpx2.Response(503, text="temporary")
            return tools.httpx2.Response(
                200,
                json={
                    "results": [
                        {"title": "T", "url": "https://example.com", "snippet": "S"}
                    ]
                },
            )

        original_client = self._use_transport(handler)
        original_delay = tools.DEEPSEEK_RETRY_DELAY_SECONDS
        tools.DEEPSEEK_RETRY_DELAY_SECONDS = 0
        try:
            result = asyncio.run(search_web("q", api_key="k"))
        finally:
            tools.DEEPSEEK_RETRY_DELAY_SECONDS = original_delay
            tools.httpx2.AsyncClient = original_client
        self.assertTrue(result.ok)

    def test_deepseek_url_does_not_duplicate_v1(self) -> None:
        self.assertEqual(
            tools._deepseek_tools_url(
                "https://api.deepseek.com/v1", tools.DEEPSEEK_SEARCH_PATH
            ),
            "https://api.deepseek.com/v1/tools/search",
        )

    def test_fetch_page_returns_markdown_and_title(self) -> None:
        captured: list[tuple[str, dict]] = []

        def handler(request):
            captured.append((str(request.url), json.loads(request.content)))
            return tools.httpx2.Response(
                200,
                json={"title": "示例页面", "content": "# 正文\n\n这是网页内容"},
            )

        original = self._use_transport(handler)
        try:
            result = asyncio.run(
                fetch_page("https://example.com/article", api_key="k", max_chars=100)
            )
        finally:
            tools.httpx2.AsyncClient = original

        self.assertTrue(result.ok)
        self.assertIn("标题：示例页面", result.output)
        self.assertIn("这是网页内容", result.output)
        self.assertEqual(captured[0][0], "https://api.deepseek.com/v1/tools/fetch")
        self.assertEqual(
            captured[0][1], {"url": "https://example.com/article"}
        )

    def test_fetch_page_reports_not_found(self) -> None:
        original = self._use_transport(
            lambda request: tools.httpx2.Response(404, text="not found")
        )
        try:
            result = asyncio.run(fetch_page("https://example.com/missing", api_key="k"))
        finally:
            tools.httpx2.AsyncClient = original
        self.assertFalse(result.ok)
        self.assertIn("没有找到可用的网页正文", result.error)

    def test_fetch_page_without_key_is_not_configured(self) -> None:
        result = asyncio.run(fetch_page("https://example.com/article"))
        self.assertFalse(result.ok)
        self.assertIn("没有配置", result.error)
        self.assertIn("LLM_API_KEY", result.error)

    def test_fetch_page_rejects_private_host(self) -> None:
        result = asyncio.run(fetch_page("http://127.0.0.1/anything"))
        self.assertFalse(result.ok)
        self.assertIn("内网", result.error)

    def test_format_search_result_includes_source_date_and_authority(self) -> None:
        result = ToolResult(
            ok=True,
            data=[
                {
                    "title": "腹泻常见原因",
                    "url": "https://example.com/a",
                    "snippet": "摘要",
                    "site_name": "example.com",
                    "date": "2026-10-20T12:00:00",
                    "authority": 4,
                }
            ],
        )
        text = MaodouClient._format_search_result(result)
        self.assertIn("来源：example.com", text)
        self.assertIn("2026-10-20", text)
        self.assertIn("权威度：高", text)

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
