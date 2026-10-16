"""表情：群里见过的表情要记得住，模型写的 [头秃] 出站时要变成真表情。"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from xiaoxia.bot import XiaoxiaClient
from xiaoxia.stickers import (
    FaceToken,
    face_book,
    face_names_in,
    face_prompt,
    face_tokens,
    render_faces,
)
from xiaoxia.store import ChatStore


GROUP = "test-group"

# 都是线上真实抓到的标签：小表情 267 是「头秃」，大表情靠附件发图，faceId 恒为 0。
SMALL_FACE = '<faceType=1,faceId="267",ext="eyJ0ZXh0Ijoi5aS056eDIn0=">'
BIG_STICKER = '<faceType=6,faceId="0",ext="eyJ0ZXh0IjoiW+Wkp+iDhl0ifQ==">'


class FaceTokenTests(unittest.TestCase):
    def test_parses_names_from_real_tags(self) -> None:
        tokens = face_tokens(f"今天好累{SMALL_FACE}你看{BIG_STICKER}")
        self.assertEqual([token.name for token in tokens], ["头秃", "大胆"])
        self.assertEqual(tokens[0].face_id, "267")
        self.assertFalse(tokens[0].is_sticker)
        self.assertTrue(tokens[1].is_sticker)

    def test_tag_round_trips(self) -> None:
        token = face_tokens(SMALL_FACE)[0]
        self.assertEqual(token.tag, SMALL_FACE)

    def test_book_prefers_the_small_face(self) -> None:
        sticker = FaceToken(name="头秃", face_type=6, face_id="0", ext="x")
        small = FaceToken(name="头秃", face_type=1, face_id="267", ext="y")
        self.assertIs(face_book([sticker, small])["头秃"], small)
        self.assertIs(face_book([small, sticker])["头秃"], small)


class RenderFacesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.book = {"头秃": FaceToken(name="头秃", face_type=1, face_id="267")}

    def test_known_name_becomes_a_real_face(self) -> None:
        self.assertEqual(render_faces("本豆要秃了[头秃]", self.book), '本豆要秃了<faceType=1,faceId="267">')

    def test_typo_is_corrected_to_the_real_name(self) -> None:
        # 线上真实翻车：它写的是 [秃头]，名字库里根本没有
        self.assertEqual(render_faces("[秃头]", self.book), '<faceType=1,faceId="267">')
        # 关掉标签通道也不能留着查无此表的名字
        self.assertEqual(render_faces("[秃头]", self.book, as_tags=False), "[头秃]")

    def test_unknown_brackets_are_left_alone(self) -> None:
        self.assertEqual(render_faces("看这个[神秘代码]哦", self.book), "看这个[神秘代码]哦")
        self.assertEqual(render_faces("[[1]] 是 markdown", self.book), "[[1]] 是 markdown")

    def test_sticker_names_stay_readable(self) -> None:
        book = face_book(face_tokens(BIG_STICKER))
        # 大表情没有图片就发不出去，纯文本里至少留个可读名字
        self.assertEqual(render_faces("走你[大胆]", book), "走你[大胆]")

    def test_sticker_names_are_not_correction_targets(self) -> None:
        # 线上真抓到过大表情名字长成 "/笑哭"，拿它去纠 [哭] 会把字改花
        book = {
            "头秃": FaceToken(name="头秃", face_type=1, face_id="267"),
            "/笑哭": FaceToken(name="/笑哭", face_type=6, face_id="0"),
        }
        self.assertEqual(render_faces("[哭]", book), "[哭]")
        self.assertEqual(render_faces("[秃头]", book), '<faceType=1,faceId="267">')

    def test_names_in_reply(self) -> None:
        self.assertEqual(face_names_in("哈哈[秃头]再说[头秃]", self.book), ["头秃"])
        self.assertEqual(face_names_in("没表情", self.book), [])

    def test_prompt_lists_names_and_is_empty_without_any(self) -> None:
        prompt = face_prompt(self.book)
        self.assertIn("[头秃]", prompt)
        self.assertEqual(face_prompt({}), "")

    def test_prompt_only_advertises_sendable_faces(self) -> None:
        book = face_book(face_tokens(f"好累{SMALL_FACE}{BIG_STICKER}"))
        prompt = face_prompt(book)
        self.assertIn("[头秃]", prompt)
        self.assertNotIn("[大胆]", prompt)
        # 只剩大表情时，等于没有能发的表情，别拿它们去钓模型
        self.assertEqual(face_prompt(face_book(face_tokens(BIG_STICKER))), "")


class FaceLearningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "faces.sqlite3"))

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _save(self, content: str) -> None:
        self.store.save_message(
            message_id=f"m-{content}",
            msg_seq=None,
            group_openid=GROUP,
            member_openid="openid",
            username="CL",
            role="member",
            content=content,
            timestamp="2026-10-16T20:00:00+08:00",
            source="GROUP_MESSAGE_CREATE",
            raw={},
        )

    def test_message_saves_the_faces_it_used(self) -> None:
        self._save(f"好累{SMALL_FACE}")
        book = self.store.face_book(GROUP)
        self.assertEqual(list(book), ["头秃"])
        self.assertEqual(book["头秃"].face_id, "267")

    def test_hits_go_up_and_names_survive_a_repeat(self) -> None:
        self._save(f"啊{SMALL_FACE}")
        self._save(f"又{SMALL_FACE}")
        row = self.store._conn.execute(
            "SELECT hits FROM face_tokens WHERE group_openid = ? AND name = ?", (GROUP, "头秃")
        ).fetchone()
        self.assertEqual(int(row["hits"]), 2)
        self.assertEqual(self.store.face_names(GROUP), ["头秃"])

    def test_plain_text_learns_nothing(self) -> None:
        self._save("今天天气不错")
        self.assertEqual(self.store.face_book(GROUP), {})


class FaceBackfillTests(unittest.TestCase):
    """老库升级：历史消息里的表情名得能补出来，不然群里用过的表情一个也发不出去。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / "legacy.sqlite3")
        self.store = ChatStore(self.path)

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _insert_legacy(self, key: str, content: str, raw_json: str) -> None:
        """老库的样子：content 已经是 [头秃]，raw_json 里还留着原始标签，face_tokens 空。"""
        conn = sqlite3.connect(self.path)
        conn.execute(
            "INSERT INTO messages (message_key, message_id, group_openid, member_openid,"
            " username, role, content, timestamp, source, raw_json)"
            " VALUES (?, ?, ?, 'openid', 'CL', 'member', ?, '2026-10-16T20:00:00+08:00',"
            " 'GROUP_MESSAGE_CREATE', ?)",
            (key, key, GROUP, content, raw_json),
        )
        conn.commit()
        conn.close()

    def _reopen(self) -> None:
        self.store.close()
        self.store = ChatStore(self.path)

    def test_history_hands_the_groups_faces_back(self) -> None:
        self._insert_legacy("m1", "好累[头秃]", json.dumps({"content": f"好累{SMALL_FACE}"}))
        self._insert_legacy("m2", "又[头秃]", json.dumps({"content": f"又{SMALL_FACE}"}))
        self._insert_legacy("m3", "就这", json.dumps({"content": "就这"}))
        self._reopen()
        book = self.store.face_book(GROUP)
        self.assertEqual(list(book), ["头秃"])
        self.assertEqual(book["头秃"].tag, SMALL_FACE)

    def test_backfill_counts_each_face_once(self) -> None:
        self._insert_legacy("m1", "好累[头秃]", json.dumps({"content": f"好累{SMALL_FACE}"}))
        self._reopen()
        self._reopen()
        row = self.store._conn.execute(
            "SELECT hits FROM face_tokens WHERE group_openid = ?", (GROUP,)
        ).fetchone()
        self.assertEqual(int(row["hits"]), 1)

    def test_unreadable_raw_json_is_skipped_quietly(self) -> None:
        self._insert_legacy("m1", "好累[头秃]", "{坏掉的 faceType")
        self._insert_legacy("m2", "又[头秃]", json.dumps({"content": f"又{SMALL_FACE}"}))
        self._reopen()
        self.assertEqual(list(self.store.face_book(GROUP)), ["头秃"])

    def test_live_learning_does_not_need_the_backfill(self) -> None:
        self._insert_legacy("m1", "好累[头秃]", json.dumps({"content": f"好累{SMALL_FACE}"}))
        self._reopen()
        self.assertEqual(self.store.backfill_face_tokens(GROUP), 0)


class OutgoingFaceTests(unittest.TestCase):
    """出站那条链路：群里看到的是表情，库里留的是可读名字。"""

    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "out.sqlite3"))
        self.store.save_message(
            message_id="seed",
            msg_seq=None,
            group_openid=GROUP,
            member_openid="openid",
            username="CL",
            role="member",
            content=f"好累{SMALL_FACE}",
            timestamp="2026-10-16T20:00:00+08:00",
            source="GROUP_MESSAGE_CREATE",
            raw={},
        )
        self.sent: list[str] = []
        self.remembered: list[str] = []
        self.client = XiaoxiaClient(
            settings=SimpleNamespace(timezone="Asia/Shanghai", face_tags=True),
            store=self.store,
            llm=SimpleNamespace(),
        )

        async def post_group_message(**kwargs):
            self.sent.append(kwargs["content"])

        self.client.api = SimpleNamespace(post_group_message=post_group_message)
        self.client._remember_bot_reply = (
            lambda group, text, source: self.remembered.append(text)
        )

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()
        asyncio.set_event_loop(None)
        self.loop.close()

    def test_reply_goes_out_as_a_face_but_is_stored_readable(self) -> None:
        message = SimpleNamespace(group_openid=GROUP, id="m-1")
        ok = self.loop.run_until_complete(
            self.client._send_reply(message, "本豆要秃了[头秃]", 1, "test")
        )
        self.assertTrue(ok)
        # 用学到的原标签发（连 ext 一起带上），格式和 QQ 自己发来的一模一样
        self.assertEqual(self.sent, [f"本豆要秃了{SMALL_FACE}"])
        # 库里必须是可读名字：标签回流进上下文就等于让它读乱码
        self.assertEqual(self.remembered, ["本豆要秃了[头秃]"])

    def test_tags_can_be_switched_off(self) -> None:
        self.client.settings = SimpleNamespace(timezone="Asia/Shanghai", face_tags=False)
        self.assertEqual(self.client._outgoing_text(GROUP, "[秃头]"), "[头秃]")


if __name__ == "__main__":
    unittest.main()
