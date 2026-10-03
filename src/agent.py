import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from browser_use import Agent, Browser, ChatOpenAI

from models import WebsiteReport
from prompts import DASHBOARD_DISCOVERY_TASK

load_dotenv()

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


async def main():
    username = os.getenv("WEBSITE_USERNAME")
    password = os.getenv("WEBSITE_PASSWORD")

    if not username or not password:
        raise ValueError("Missing login credentials in .env")

    llm = ChatOpenAI(
        model="qwen3.5-9b-mtp",
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
    )

    # Deterministic actions executed before the first LLM call
    initial_actions = [
        {
            "navigate": {
                "url": "https://dev-task.srivyn.in/login",
                "new_tab": False,
            }
        }
    ]

    browser = Browser(
        headless=False,
        window_size={"width": 1440, "height": 900},
    )

    agent = Agent(
        task=DASHBOARD_DISCOVERY_TASK,
        llm=llm,
        browser=browser,
        sensitive_data={
            "x_username": os.getenv("WEBSITE_USERNAME", ""),
            "x_password": os.getenv("WEBSITE_PASSWORD", ""),
        },
        output_model_schema=WebsiteReport,
    )

    try:
        history = await agent.run(max_steps=50)

        # Get the structured Pydantic result.
        report = history.structured_output

        if report:
            output_file = OUTPUT_DIR / "discovery.json"
            output_file.write_text(
                report.model_dump_json(indent=2),
                encoding="utf-8",
            )
            print(f"Discovery report saved to {output_file}")
        else:
            print("No structured result returned.")
            print("Errors:", history.errors())

    finally:
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())