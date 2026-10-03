
import asyncio

from browser_use import Agent, ChatOpenAI


async def main():
    llm = ChatOpenAI(
        model="qwen3.5-9b-mtp",
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
    )

    agent = Agent(
        task="Open https://dev-task.srivyn.in/login and tell me the page title.",
        llm=llm,
        use_thinking=False
    )

    history = await agent.run(max_steps=5)
    print(history)


if __name__ == "__main__":
    asyncio.run(main())