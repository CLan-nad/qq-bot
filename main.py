from __future__ import annotations

import asyncio
import logging

from maodou.bot import MaodouClient
from maodou.config import load_settings
from maodou.llm import MaodouLLM
from maodou.store import ChatStore


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger("maodou").setLevel(logging.INFO)
    logging.getLogger("botpy").setLevel(logging.WARNING)

    settings = load_settings()
    store = ChatStore(settings.db_path)
    llm = MaodouLLM(settings)

    # qq-botpy 1.2.1 expects an event loop to exist before Client.__init__.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    client = MaodouClient(settings=settings, store=store, llm=llm)
    try:
        client.run(appid=settings.qq_app_id, secret=settings.qq_app_secret)
    finally:
        store.close()


if __name__ == "__main__":
    main()

