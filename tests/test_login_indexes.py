import asyncio
from browser_use import BrowserSession


async def main():
    browser = BrowserSession()
    await browser.start()

    try:
        page = await browser.get_current_page()
        await page.goto("https://dev-task.srivyn.in/login")

        state = await browser.get_browser_state_summary()

        for index, element in state.dom_state.selector_map.items():
            tag = element.node_name
            attrs = element.attributes or {}

            print(
                f"Index: {index} | "
                f"Tag: {tag} | "
                f"Attributes: {attrs}"
            )

    finally:
        await browser.stop()


if __name__ == "__main__":
    asyncio.run(main())