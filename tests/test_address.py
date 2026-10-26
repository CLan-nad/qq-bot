"""称呼（怎么叫某个人）这条链路：存、归一、注入身份块、群里下指令改。

称呼属于人设，代码里不写死任何一个；这里断言的全是“登记了就一定照做”。
"""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from maodou.bot import IdentityContext, MaodouClient
from maodou.llm import INTENT_SYSTEM_PROMPT, SYSTEM_PROMPT, TOOL_REPLY_ADDENDUM
from maodou.store import ChatStore, StoredMessage


GROUP = "test-group"


def _member(username: str, role: str = "member") -> StoredMessage:
    return StoredMessage(
        username=username,
        member_openid=f"openid-{username}",
        role=role,
        content="",
        timestamp="2026-10-14T15:00:00+08:00",
        source="GROUP_MESSAGE_CREATE",
    )


class PersonAddressStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "addr.sqlite3"))

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_set_get_overwrite_clear(self) -> None:
        self.assertEqual(self.store.address_for(GROUP, "CL"), "")
        self.assertEqual(self.store.set_person_address(GROUP, "CL", "主人"), "CL")
        self.assertEqual(self.store.address_for(GROUP, "CL"), "主人")
        # 改口不新增一行，直接覆盖
        self.store.set_person_address(GROUP, "CL", "老板")
        self.assertEqual(self.store.person_addresses(GROUP), {"CL": "老板"})
        self.assertTrue(self.store.clear_person_address(GROUP, "CL"))
        self.assertEqual(self.store.address_for(GROUP, "CL"), "")
        self.assertFalse(self.store.clear_person_address(GROUP, "CL"))

    def test_blank_values_are_refused(self) -> None:
        self.assertEqual(self.store.set_person_address(GROUP, "", "主人"), "")
        self.assertEqual(self.store.set_person_address(GROUP, "CL", "  "), "")
        self.assertEqual(self.store.person_addresses(GROUP), {})
        # 引号是模型偶尔会连着一起吐出来的，落库前脱掉
        self.store.set_person_address(GROUP, "CL", "「主人」")
        self.assertEqual(self.store.address_for(GROUP, "CL"), "主人")

    def test_address_is_stored_under_the_canonical_name(self) -> None:
        self.store.set_person_alias(GROUP, "二十四桥明月夜", "小虾")
        self.store.set_person_address(GROUP, "二十四桥明月夜", "小虾主人")
        # 落库的是归一后的名字，用哪个叫法问都问得到
        self.assertEqual(self.store.person_addresses(GROUP), {"小虾": "小虾主人"})
        self.assertEqual(self.store.address_for(GROUP, "二十四桥明月夜"), "小虾主人")
        self.assertEqual(self.store.address_for(GROUP, "小虾"), "小虾主人")

    def test_alias_merges_carry_the_address_over(self) -> None:
        # 先定称呼、后统一叫法：称呼要跟着人搬到新叫法名下，不能丢。
        self.store.set_person_address(GROUP, "二十四桥明月夜", "小虾主人")
        self.store.set_person_alias(GROUP, "二十四桥明月夜", "小虾")
        self.assertEqual(self.store.person_addresses(GROUP), {"小虾": "小虾主人"})
        self.assertEqual(self.store.address_for(GROUP, "二十四桥明月夜"), "小虾主人")

    def test_alias_merge_keeps_the_canonical_address(self) -> None:
        # 两边都定过称呼时，归一后保留新叫法自己那一份。
        self.store.set_person_address(GROUP, "二十四桥明月夜", "老称呼")
        self.store.set_person_address(GROUP, "小虾", "小虾主人")
        self.store.set_person_alias(GROUP, "二十四桥明月夜", "小虾")
        self.assertEqual(self.store.person_addresses(GROUP), {"小虾": "小虾主人"})


class IdentityPromptTests(unittest.TestCase):
    def test_prompt_lists_addresses(self) -> None:
        identity = IdentityContext(
            requester=_member("CL", "owner"),
            partner=_member("二十四桥明月夜"),
            addresses={"CL": "主人", "二十四桥明月夜": "小虾主人"},
        )
        prompt = identity.prompt()
        self.assertIn("当前触发者：CL，请叫「主人」", prompt)
        self.assertIn("另一位聊天成员：二十四桥明月夜，请叫「小虾主人」", prompt)
        self.assertIn("称呼要求（他们自己定的，每一轮都要照做）", prompt)
        self.assertIn("CL → 「主人」", prompt)
        self.assertIn("二十四桥明月夜 → 「小虾主人」", prompt)

    def test_prompt_without_addresses_stays_as_before(self) -> None:
        identity = IdentityContext(requester=_member("CL"), partner=None)
        prompt = identity.prompt()
        self.assertNotIn("称呼要求", prompt)
        self.assertIn("当前触发者：CL（成员）", prompt)
        self.assertIn("另一位聊天成员：暂时未识别", prompt)


class BotAddressTests(unittest.TestCase):
    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "bot-addr.sqlite3"))
        self.client = MaodouClient(
            settings=SimpleNamespace(timezone="Asia/Shanghai"),
            store=self.store,
            llm=SimpleNamespace(),
        )

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()
        asyncio.set_event_loop(None)
        self.loop.close()

    def _save(self, content: str, username: str) -> None:
        self.store.save_message(
            message_id=f"id-{content}",
            msg_seq="",
            group_openid=GROUP,
            member_openid=f"openid-{username}",
            username=username,
            role="member",
            content=content,
            timestamp="2026-10-14T15:00:00+08:00",
            source="GROUP_MESSAGE_CREATE",
            raw={},
        )

    def _identity(self, username: str = "CL") -> IdentityContext:
        return IdentityContext(requester=_member(username), partner=_member("二十四桥明月夜"))

    def test_identity_context_resolves_addresses(self) -> None:
        self._save("第一条", "CL")
        self._save("第二条", "二十四桥明月夜")
        self.store.set_person_address(GROUP, "CL", "主人")
        self.store.set_person_alias(GROUP, "二十四桥明月夜", "小虾")
        self.store.set_person_address(GROUP, "小虾", "小虾主人")

        members = {member.username: member for member in self.store.known_members(GROUP)}
        identity = self.client._identity_context(GROUP, members["CL"])
        # 显示名还是老叫法，但称呼按归一后的名字查得到
        self.assertEqual(
            identity.addresses, {"CL": "主人", "二十四桥明月夜": "小虾主人"}
        )
        self.assertIn("「小虾主人」", identity.prompt())

    def test_handle_address_set_show_set_clear(self) -> None:
        reply = self.client._handle_address_set({"op": "show"}, SimpleNamespace(
            group_openid=GROUP, username="CL"), self._identity())
        self.assertIn("还没定过称呼", reply)

        reply = self.client._handle_address_set(
            {"op": "set", "person": "CL", "address": "主人"},
            SimpleNamespace(group_openid=GROUP, username="CL"),
            self._identity(),
        )
        self.assertIn("以后就叫CL「主人」", reply)
        self.assertEqual(self.store.address_for(GROUP, "CL"), "主人")

        reply = self.client._handle_address_set({"op": "show"}, SimpleNamespace(
            group_openid=GROUP, username="CL"), self._identity())
        self.assertIn("CL → 「主人」", reply)

        reply = self.client._handle_address_set(
            {"op": "clear", "person": "CL"},
            SimpleNamespace(group_openid=GROUP, username="CL"),
            self._identity(),
        )
        self.assertIn("以后就直接叫CL了", reply)
        self.assertEqual(self.store.address_for(GROUP, "CL"), "")

    def test_handle_address_set_defaults_to_the_requester(self) -> None:
        # “以后叫我主人”里没说 person，默认就是当前 @ 我的这个人
        reply = self.client._handle_address_set(
            {"op": "set", "address": "主人"},
            SimpleNamespace(group_openid=GROUP, username="CL"),
            self._identity(),
        )
        self.assertIn("以后就叫CL「主人」", reply)
        self.assertEqual(self.store.address_for(GROUP, "CL"), "主人")

    def test_handle_address_set_needs_an_address(self) -> None:
        reply = self.client._handle_address_set(
            {"op": "set", "person": "CL"},
            SimpleNamespace(group_openid=GROUP, username="CL"),
            self._identity(),
        )
        self.assertIn("把称呼说出来", reply)
        self.assertEqual(self.store.person_addresses(GROUP), {})

    def test_handle_address_set_clear_unknown_person(self) -> None:
        reply = self.client._handle_address_set(
            {"op": "clear", "person": "路人"},
            SimpleNamespace(group_openid=GROUP, username="CL"),
            self._identity(),
        )
        self.assertIn("本来就没定过称呼", reply)

    def test_dispatch_routes_address_set(self) -> None:
        reply = self.loop.run_until_complete(
            self.client._handle_intent(
                intent={"action": "address_set", "op": "show"},
                message=SimpleNamespace(group_openid=GROUP, username="CL"),
                requester=_member("CL"),
                request="你现在怎么称呼我们",
                topic="",
                background="",
                full_recent="",
                identity=self._identity(),
            )
        )
        self.assertIn("还没定过称呼", reply)


class PromptRuleTests(unittest.TestCase):
    def test_system_prompt_has_address_rules(self) -> None:
        self.assertIn("5. 称呼一律照", SYSTEM_PROMPT)
        self.assertIn("不要用显示名代替", SYSTEM_PROMPT)
        self.assertIn("称呼只是叫法，不是事实", SYSTEM_PROMPT)

    def test_tool_addendum_numbers_follow_the_main_rules(self) -> None:
        # 主规则 7 条，工具补充规则必须从 8 顺延，两处不能出现同一个编号
        self.assertIn("8. 忠于工具结果", TOOL_REPLY_ADDENDUM)
        self.assertIn("11. 这是简单查询", TOOL_REPLY_ADDENDUM)
        self.assertIn("不要为了显得详细硬凑成长报告", TOOL_REPLY_ADDENDUM)
        self.assertNotIn("5. 称呼", TOOL_REPLY_ADDENDUM)
        self.assertNotIn("7. 忠于", TOOL_REPLY_ADDENDUM)

    def test_intent_prompt_describes_address_set(self) -> None:
        self.assertIn("- address_set:", INTENT_SYSTEM_PROMPT)
        for field in ("address_set 输出 op", "address 只填称呼本身", "person 填当前触发者的显示名"):
            self.assertIn(field, INTENT_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
