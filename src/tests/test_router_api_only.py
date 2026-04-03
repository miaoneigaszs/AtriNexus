import json
import os

import httpx


API_KEY_ENV = "ATRINEXUS_ROUTER_TEST_API_KEY"
BASE_URL_ENV = "ATRINEXUS_ROUTER_TEST_BASE_URL"
MODEL_ENV = "ATRINEXUS_ROUTER_TEST_MODEL"


def run_router_smoke_test() -> None:
    api_key = os.getenv(API_KEY_ENV, "").strip()
    if not api_key:
        raise RuntimeError(
            f"Missing API key. Set {API_KEY_ENV} before running this script."
        )

    base_url = os.getenv(BASE_URL_ENV, "https://api.siliconflow.cn/v1").strip()
    model = os.getenv(MODEL_ENV, "Qwen/Qwen2.5-7B-Instruct").strip()

    content = "那个流程是什么？"
    previous_context = "用户刚才一直在问关于财务报销的事情。"

    system_prompt = (
        "你是一个工业级 RAG 意图路由器核心。用户输入了一句话，请判断他是否需要查询外部知识库资料。\n"
        "判断标准：\n"
        "1. 询问公司规章、指南、手册、流程等客观知识 -> TYPE_KNOWLEDGE_BASE\n"
        "2. 普通打招呼、讲笑话、表达情绪、非求知指令 -> TYPE_CHITCHAT\n"
        "如果判断为 TYPE_KNOWLEDGE_BASE，你必须结合下方的【最近聊天上下文】，生成一个消除所有代词的独立查询句。\n"
        "比如上下文讲过报销，用户只发了“机票呢”，独立查询句应为“报销规定中关于机票的标准”。\n"
        "请严格以 JSON 格式输出，示例: "
        "{\"intent\": \"TYPE_KNOWLEDGE_BASE\", \"query\": \"重写后的完整提问\"} "
        "或 {\"intent\": \"TYPE_CHITCHAT\"}。\n"
        "【最近聊天上下文】:\n"
        f"{previous_context[-500:]}"
    )

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 100,
        "temperature": 0.1,
    }

    with httpx.Client(timeout=30) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        reply = result["choices"][0]["message"]["content"].strip()
        print(json.dumps({"reply": reply}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_router_smoke_test()
