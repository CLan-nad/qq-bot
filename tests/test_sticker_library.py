"""表情包库：assets/stickers/ 里那批图怎么被读出来、怎么被摘出回复。

小表情的标签通道在 QQ 群消息里走不通（线上实测只剩一对尖括号），所以小毛豆想发表情
只能发图。图是本地素材，模型眼里只有一个名字：写 [抱抱]，出站时摘出来换成图片。
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from maodou.stickers import StickerLibrary


GIF_BYTES = b"GIF89a-not-really-a-gif"


class StickerLibraryFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "a_1f917.gif").write_bytes(GIF_BYTES)
        (self.root / "a_1f970.gif").write_bytes(b"GIF89a-second")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_manifest(self, items: list[dict] | None = None, *, raw: str | None = None) -> None:
        payload = raw if raw is not None else json.dumps({"version": 1, "items": items or []})
        (self.root / "manifest.json").write_text(payload, encoding="utf-8")

    def _standard(self) -> StickerLibrary:
        self._write_manifest(
            [
                {"name": "抱抱", "file": "a_1f917.gif", "emoji": "🤗", "tags": ["抱抱", "撒娇"]},
                {"name": "爱心眼", "file": "a_1f970.gif", "emoji": "🥰", "tags": []},
            ]
        )
        return StickerLibrary.load(self.root)


class StickerLoadTests(StickerLibraryFixture):
    def test_loads_names_emoji_tags_and_files(self) -> None:
        library = self._standard()
        self.assertTrue(library)
        self.assertEqual(len(library), 2)
        self.assertEqual(library.names(), ["抱抱", "爱心眼"])

        sticker = library.get("抱抱")
        self.assertIsNotNone(sticker)
        assert sticker is not None
        self.assertEqual(sticker.name, "抱抱")
        self.assertEqual(sticker.emoji, "🤗")
        self.assertEqual(sticker.tags, ("抱抱", "撒娇"))
        self.assertEqual(sticker.content_type, "image/gif")
        self.assertEqual(sticker.read(), GIF_BYTES)

    def test_missing_manifest_is_an_empty_library(self) -> None:
        # 素材目录没跟着代码走（比如只在本地有）时，机器人得照常起来
        library = StickerLibrary.load(self.root)
        self.assertFalse(library)
        self.assertEqual(library.names(), [])
        self.assertIsNone(library.get("抱抱"))

    def test_broken_manifest_is_an_empty_library(self) -> None:
        self._write_manifest(raw="{不是合法 json")
        library = StickerLibrary.load(self.root)
        self.assertFalse(library)
        self.assertEqual(library.take("好耶[抱抱]"), ("好耶[抱抱]", None))

    def test_item_pointing_at_a_missing_file_is_skipped(self) -> None:
        self._write_manifest(
            [
                {"name": "抱抱", "file": "a_1f917.gif", "emoji": "🤗", "tags": []},
                {"name": "幽灵", "file": "nope.gif", "emoji": "👻", "tags": []},
                {"name": "", "file": "a_1f970.gif", "emoji": "", "tags": []},
                "not-a-dict",
            ]
        )
        library = StickerLibrary.load(self.root)
        self.assertEqual(library.names(), ["抱抱"])

    def test_missing_root_directory_is_an_empty_library(self) -> None:
        library = StickerLibrary.load(self.root / "不存在的目录")
        self.assertFalse(library)


class StickerTakeTests(StickerLibraryFixture):
    def setUp(self) -> None:
        super().setUp()
        self.library = self._standard()

    def test_sticker_name_is_pulled_out_of_the_reply(self) -> None:
        remainder, sticker = self.library.take("抱抱你呀[抱抱]")
        self.assertEqual(remainder, "抱抱你呀")
        self.assertIsNotNone(sticker)
        assert sticker is not None
        self.assertEqual(sticker.name, "抱抱")

    def test_only_the_first_name_becomes_an_image(self) -> None:
        # 一条回复最多发一张图：QQ 每张图都是一次上传加一条消息，发多了刷屏
        remainder, sticker = self.library.take("开心[爱心眼]又[抱抱]")
        self.assertEqual(remainder, "开心又[抱抱]")
        self.assertEqual(sticker.name, "爱心眼")

    def test_a_reply_that_is_only_a_name_leaves_no_text(self) -> None:
        # 正文空掉是合法的：调用方会把它当「这条只发图」处理
        remainder, sticker = self.library.take("[抱抱]")
        self.assertEqual(remainder, "")
        self.assertEqual(sticker.name, "抱抱")

    def test_spaces_left_behind_are_tidied_up(self) -> None:
        remainder, _ = self.library.take("好耶 [抱抱] 抱一下")
        self.assertEqual(remainder, "好耶 抱一下")

    def test_unknown_names_and_plain_words_are_left_alone(self) -> None:
        self.assertEqual(self.library.take("想[抱抱你]"), ("想[抱抱你]", None))
        # 名字不带方括号就只是普通文字，不能被摘走
        self.assertEqual(self.library.take("我想抱抱你"), ("我想抱抱你", None))
        self.assertEqual(self.library.take("[[1]] 是 markdown"), ("[[1]] 是 markdown", None))

    def test_no_fuzzy_matching_on_sticker_names(self) -> None:
        # 纠错是给小表情的（[秃头] → [头秃]）；套在图名上容易把一句好话改成别的表情
        self.assertEqual(self.library.take("抱一下[抱抱你]"), ("抱一下[抱抱你]", None))

    def test_empty_library_passes_everything_through(self) -> None:
        empty = StickerLibrary(self.root, {})
        self.assertEqual(empty.take("抱抱你呀[抱抱]"), ("抱抱你呀[抱抱]", None))
        self.assertEqual(empty.take(""), ("", None))
        self.assertEqual(empty.take(None), ("", None))


class StickerPromptTests(StickerLibraryFixture):
    def test_prompt_lists_every_name_and_forbids_inventing(self) -> None:
        prompt = self._standard().prompt()
        self.assertIn("[抱抱]", prompt)
        self.assertIn("[爱心眼]", prompt)
        self.assertIn("🤗", prompt)
        self.assertIn("最多发一张", prompt)
        self.assertIn("别自己造", prompt)

    def test_empty_library_advertises_nothing(self) -> None:
        self.assertEqual(StickerLibrary(self.root, {}).prompt(), "")


if __name__ == "__main__":
    unittest.main()
