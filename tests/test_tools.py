from __future__ import annotations

import asyncio
import unittest
from io import BytesIO
from types import SimpleNamespace

import xiaoxia.tools as tools
from xiaoxia.tools import (
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
    def test_search_auto_uses_tavily_when_key_present(self) -> None:
        async def fake_tavily(query, limit, key):
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

    def test_search_falls_back_to_duckduckgo(self) -> None:
        async def fake_ddg(query, limit):
            return ToolResult(ok=True, data=[{"title": "T", "url": "u", "snippet": "s"}])

        original = tools._duckduckgo_search
        tools._duckduckgo_search = fake_ddg
        try:
            result = asyncio.run(search_web("q", provider="auto", tavily_api_key=""))
        finally:
            tools._duckduckgo_search = original
        self.assertTrue(result.ok)

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
