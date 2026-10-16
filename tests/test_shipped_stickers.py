"""随仓库走的那批表情包：清单、文件、名字都得能直接发出去。

素材装好之后就进 git 了，装错不会在本地报错，只会在群里发不出去。所以这里当一道
闸：名字必须能被 take() 摘走、文件必须真的在、体积别超上限。
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from xiaoxia.stickers import FACE_NAME_PATTERN, StickerLibrary


LIBRARY = Path(__file__).resolve().parents[1] / "assets" / "stickers"
# 单张上限：QQ 图片 20MB，这里留一大截余量，免得一个手滑把几十兆的图塞进仓库。
MAX_BYTES = 500 * 1024


class ShippedStickerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.library = StickerLibrary.load(LIBRARY)
        self.items = json.loads((LIBRARY / "manifest.json").read_text(encoding="utf-8"))["items"]

    def test_library_is_actually_loaded(self) -> None:
        self.assertGreaterEqual(len(self.library), 40)
        self.assertEqual(len(self.library), len(self.items))

    def test_every_name_survives_the_take_pattern(self) -> None:
        # 名字里带空格、标点或者超过 8 个字，模型写了也摘不出来，等于白装
        for item in self.items:
            name = item["name"]
            with self.subTest(name=name):
                self.assertTrue(FACE_NAME_PATTERN.fullmatch(f"[{name}]"), name)
                self.assertLessEqual(len(name), 8)

    def test_names_are_unique_and_typed(self) -> None:
        names = [item["name"] for item in self.items]
        self.assertEqual(len(names), len(set(names)))
        # 重名会让 take() 只认清单里靠前的那张，另一张永远发不出去
        for item in self.items:
            with self.subTest(name=item["name"]):
                self.assertTrue(item.get("file"))
                # emoji 是给模型认的标签，缺了那一行就读不懂这张图什么情绪
                self.assertTrue(item.get("emoji"))

    def test_every_name_actually_pulls_out_its_own_image(self) -> None:
        for item in self.items:
            with self.subTest(name=item["name"]):
                remainder, sticker = self.library.take(f"好耶[{item['name']}]")
                self.assertEqual(remainder, "好耶")
                self.assertIsNotNone(sticker)
                assert sticker is not None
                self.assertEqual(sticker.path.name, item["file"])

    def test_files_exist_and_stay_small(self) -> None:
        total = 0
        for item in self.items:
            path = LIBRARY / item["file"]
            with self.subTest(file=item["file"]):
                self.assertTrue(path.is_file(), item["file"])
                size = path.stat().st_size
                self.assertLessEqual(size, MAX_BYTES, f"{item['file']} 太大了")
                total += size
        # 仓库里塞太多图会把 clone 和部署一起拖慢，这里是总闸
        self.assertLessEqual(total, 30 * 1024 * 1024)

    def test_files_are_real_images(self) -> None:
        # QQ 只吃这几样：认不出来就是发不出去，而且失败发生在出站那一刻，很难查
        for item in self.items:
            head = (LIBRARY / item["file"]).read_bytes()[:8]
            with self.subTest(file=item["file"]):
                self.assertTrue(
                    head.startswith(b"GIF87a") or head.startswith(b"GIF89a") or head.startswith(b"\x89PNG"),
                    f"{item['file']} 不是 GIF/PNG",
                )

    def test_maodou_pack_is_present(self) -> None:
        # 本尊那几张是压轴：模型自称「小毛豆」的时候得有它自己的脸可发
        names = self.library.names()
        self.assertIn("本尊", names)
        self.assertIn("本尊委屈", names)

    def test_prompt_offers_both_packs(self) -> None:
        prompt = self.library.prompt()
        self.assertIn("[抱抱]", prompt)
        self.assertIn("[本尊]", prompt)


if __name__ == "__main__":
    unittest.main()
