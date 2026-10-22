"""称呼要钉在 openid 上：对方改昵称、历史里显示名变了，机器人也得照样认得出来。"""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from maodou.store import ChatStore


GROUP = "test-group"
CL_OPENID = "7224E8B8F70CB7A5219ABB066F8346C7"
PARTNER_OPENID = "B80D6818756E5BF463FF6B3D16F4124B"

# 加 member_openid 之前的老表：称呼只挂在显示名上。
OLD_ADDRESS_SCHEMA = """
CREATE TABLE person_addresses (
    group_openid TEXT NOT NULL,
    person TEXT NOT NULL,
    address TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (group_openid, person)
);
"""


class AddressOpenidTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "addr.sqlite3"))
        self._save("CL", CL_OPENID, "我在")
        self._save("二十四桥明月夜", PARTNER_OPENID, "在呢")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _save(self, username: str, member_openid: str, content: str) -> None:
        self.store.save_message(
            message_id=f"m-{username}-{content}",
            msg_seq=None,
            group_openid=GROUP,
            member_openid=member_openid,
            username=username,
            role="member",
            content=content,
            timestamp="2026-10-16T20:00:00+08:00",
            source="GROUP_MESSAGE_CREATE",
            raw={},
        )

    def test_set_address_binds_the_openid(self) -> None:
        self.assertEqual(self.store.set_person_address(GROUP, "CL", "主人"), "CL")
        self.assertEqual(self.store.address_for_member(GROUP, CL_OPENID), "主人")
        # 别人不该被连带绑上
        self.assertEqual(self.store.address_for_member(GROUP, PARTNER_OPENID), "")

    def test_nickname_change_keeps_the_address(self) -> None:
        self.store.set_person_address(GROUP, "CL", "主人")
        # 同一 openid 换了显示名
        self._save("C-L", CL_OPENID, "改名啦")
        labels = self.store.speaker_labels(GROUP)
        self.assertEqual(labels.get(CL_OPENID), "主人")
        self.assertEqual(labels.get("C-L"), "主人")
        self.assertEqual(labels.get(PARTNER_OPENID), "二十四桥明月夜")

    def test_alias_of_the_same_person_binds_too(self) -> None:
        self.store.set_person_alias(GROUP, "小虾", "二十四桥明月夜")
        self.assertEqual(self.store.set_person_address(GROUP, "小虾", "小虾主人"), "二十四桥明月夜")
        self.assertEqual(self.store.address_for_member(GROUP, PARTNER_OPENID), "小虾主人")

    def test_one_openid_keeps_only_the_newest_row(self) -> None:
        self.store.set_person_address(GROUP, "CL", "主人")
        # 对方改昵称后重新登记称呼：走别名归一到同一行，openid 不能出现在两行里
        self.store.set_person_alias(GROUP, "C-L", "CL")
        self.store.set_person_address(GROUP, "C-L", "主人")
        rows = self.store._conn.execute(
            "SELECT person FROM person_addresses WHERE group_openid = ? AND member_openid = ?",
            (GROUP, CL_OPENID),
        ).fetchall()
        self.assertEqual([str(row["person"]) for row in rows], ["CL"])
        self.assertEqual(self.store.address_for_member(GROUP, CL_OPENID), "主人")

    def test_clear_removes_the_binding(self) -> None:
        self.store.set_person_address(GROUP, "CL", "主人")
        self.assertTrue(self.store.clear_person_address(GROUP, "CL"))
        self.assertEqual(self.store.address_for_member(GROUP, CL_OPENID), "")


class AddressBackfillTests(unittest.TestCase):
    """老库迁移：补列 → 称呼自动补上 openid，且重复启动不会出错。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "old.sqlite3")
        conn = sqlite3.connect(self.db_path)
        conn.executescript(OLD_ADDRESS_SCHEMA)
        conn.execute(
            "INSERT INTO person_addresses (group_openid, person, address) VALUES (?, ?, ?)",
            (GROUP, "二十四桥明月夜", "小虾主人"),
        )
        conn.commit()
        conn.close()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_migration_backfills_openid(self) -> None:
        store = ChatStore(self.db_path)
        try:
            store.save_message(
                message_id="m-1",
                msg_seq=None,
                group_openid=GROUP,
                member_openid=PARTNER_OPENID,
                username="二十四桥明月夜",
                role="member",
                content="在呢",
                timestamp="2026-10-16T20:00:00+08:00",
                source="GROUP_MESSAGE_CREATE",
                raw={},
            )
            # 补列前的空值是 ''，_migrate 里回填成真 openid
            store.backfill_address_openids(GROUP)
            self.assertEqual(store.address_for_member(GROUP, PARTNER_OPENID), "小虾主人")
            self.assertEqual(store.address_for(GROUP, "二十四桥明月夜"), "小虾主人")
        finally:
            store.close()

        # 再开一次：迁移 + 回填都是幂等的
        reopened = ChatStore(self.db_path)
        try:
            self.assertEqual(reopened.person_addresses(GROUP), {"二十四桥明月夜": "小虾主人"})
            self.assertEqual(reopened.address_for_member(GROUP, PARTNER_OPENID), "小虾主人")
        finally:
            reopened.close()

    def test_startup_backfill_runs_by_itself(self) -> None:
        store = ChatStore(self.db_path)
        try:
            store.save_message(
                message_id="m-1",
                msg_seq=None,
                group_openid=GROUP,
                member_openid=PARTNER_OPENID,
                username="二十四桥明月夜",
                role="member",
                content="在呢",
                timestamp="2026-10-16T20:00:00+08:00",
                source="GROUP_MESSAGE_CREATE",
                raw={},
            )
            # 模拟「迁移只补了空列、还没回填」的中间状态
            store._conn.execute(
                "UPDATE person_addresses SET member_openid = '' WHERE group_openid = ?",
                (GROUP,),
            )
            store._conn.commit()
        finally:
            store.close()

        reopened = ChatStore(self.db_path)
        try:
            self.assertEqual(reopened.address_for_member(GROUP, PARTNER_OPENID), "小虾主人")
        finally:
            reopened.close()


if __name__ == "__main__":
    unittest.main()
