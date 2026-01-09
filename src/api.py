import os
from typing import Mapping, Sequence

try:
    from dotenv import load_dotenv
except ImportError:  # 允许缺少 dotenv 也能运行
    def load_dotenv() -> None:
        return None

from openai import OpenAI

# 加载环境变量
load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)


def chat_completion(
    messages: Sequence[Mapping[str, str]],
    model: str = "deepseek-chat",
    temperature: float = 0.2,
) -> str:
    """
    Call the chat completion API with custom messages.
    Returns only the assistant content string.
    """
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=list(messages),
    )
    return response.choices[0].message.content


def get_ai_response(user_input: str, model: str = "deepseek-chat") -> str:
    """
    保留原始简单对话接口。
    """
    try:
        return chat_completion(
            [
                {"role": "system", "content": "你是一个有用的助手。"},
                {"role": "user", "content": user_input},
            ],
            model=model,
        )
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    print(get_ai_response("你好"))
