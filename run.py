import os
# 导入你的 Agent 和 模型工厂
from tinyAgent.Agent import Agent
from tinyAgent.LLM import OpenAILikeChat, InternLM2Chat


def main():
    print("================ 初始化系统 ================")

    # 1. 从环境变量中读取 API Key
    # 假设我们将环境变量命名为 "MIMO_API_KEY"
    API_KEY = os.getenv("MIMO_API_KEY")

    # 加上一个安全校验：如果没读到，抛出明确的错误提示，防止后面瞎报错
    if not API_KEY:
        raise ValueError("🚨 错误：未找到环境变量 'MIMO_API_KEY'，请先配置！")

    BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
    MODEL_NAME = "mimo-v2.5-pro"

    # 2. 组装引擎
    # 【方案 A：使用 API 引擎】
    engine = OpenAILikeChat(
        api_key=API_KEY,
        base_url=BASE_URL,
        model_name=MODEL_NAME
    )

    # 【方案 B：使用本地模型引擎】
    # 如果以后你有显卡了，直接把上面注释掉，把下面这行取消注释即可：
    # engine = InternLM2Chat('/root/share/model_repos/internlm2-chat-7b')

    # 3. 注入引擎，启动 Agent！
    agent = Agent(model_engine=engine)
    print("================ 初始化完成 ================\n")

    # # ===================================================
    # # 测试模式 1：单次提问测试（你刚才发的版本）
    # # ===================================================
    # print(">>> 【测试模式 1：单次任务】")
    # test_prompt = "请帮我查一下明天的天气"
    # print(f"[User]: {test_prompt}")
    #
    # # 注意：运行这步时，Agent 内部会打印出思考和调用插件的过程
    # response, _ = agent.text_completion(test_prompt, history=[])
    # print(f"\n[最终回复]: {response}")
    # print("-" * 50)

    # ===================================================
    # 测试模式 2：连续交互对话（强推！体验更好）
    # ===================================================
    print("\n>>> 【测试模式 2：连续对话】 (输入 'quit' 或 'exit' 退出)")
    chat_history = []  # 用来保存多轮对话的记忆

    while True:
        user_input = input("\n[User]: ")

        # 退出机制
        if user_input.lower() in ['quit', 'exit']:
            print("Agent 已退出，再见！")
            break

        # 防止输入空字符
        if not user_input.strip():
            continue

        # 调用 Agent，并更新历史记录
        response, chat_history = agent.text_completion(user_input, history=chat_history)

        print(f"\n[Agent]: {response}")
        print(f"[Chat_history]: {chat_history}")


if __name__ == "__main__":
    main()