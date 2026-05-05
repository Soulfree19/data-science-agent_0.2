import os
from langchain_openai import ChatOpenAI


def create_llm() -> ChatOpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = (
        os.getenv("DEEPSEEK_BASE_URL")
        or os.getenv("DEEPSEEK_API_BASE")
        or "https://api.deepseek.com"
    )

    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY 未设置，请检查 .env 文件")

    return ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url=base_url,
        max_tokens=4096,
        temperature=0,
        streaming=True,
    )
