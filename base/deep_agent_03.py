# _*_ coding : utf-8 _*_
# @Time : 2026/10/19 14:00
# @Author : KarryLiu
# File : deep_agent_03
# @Project : DeepAgent
import asyncio
import os

from deepagents import create_deep_agent
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
from langchain.chat_models import init_chat_model

llm = init_chat_model(
    model=os.getenv("LLM_DEEPSEEK"),
    model_provider="openai"
)

"""
天气助手 查询天气（固定返回“晴朗”）。
计算助手 处理数学问题。
翻译助手 负责中英互译。
"""
# 天气助手 查询天气（固定返回“晴朗”）。
weather_agent = {
    "name": "weather_helper",
    "description": "用于查询天气信息。当用户询问天气时，请调用此助手。",
    "system_prompt": "你是一个天气助手。无论用户问哪个城市的天气，你都统一回答：'今日天气晴朗，气温 25 度，适合出游。'",
    "tools": []  # 这里不需要额外工具，仅靠 prompt 回复
}

# 计算助手 处理数学问题。
calculate_agent = {
    "name": "calculate_helper",
    "description": "用于处理数学问题。当用户询问数学计算时，请调用此助手。",
    "system_prompt": "你是一个计算助手。请根据用户提供的数学问题，给出详细的解答步骤和最终答案。",
    "tools": []  # 这里不需要额外工具，仅靠 prompt 回复
}

# 翻译助手 负责中英互译。
translate_agent = {
    "name": "translate_helper",
    "description": "用于中英文翻译。当用户需要翻译时，请调用此助手。",
    "system_prompt": "你是一个翻译助手。请根据用户提供的文本，进行中英文互译，并保持原意。",
    "tools": []  # 这里不需要额外工具，仅靠 prompt 回复
}

# 创建主智能体，一主三从

main_agent = create_deep_agent(
    model=llm,
    name="main_agent",
    subagents=[weather_agent, calculate_agent, translate_agent],
    system_prompt="这是一个主智能体，负责协调天气助手、计算助手和翻译助手的调用。",
)


def test_stream(query):
    """
    测试主智能体的功能
    :param query: 用户输入的查询
    :return: 主智能体的响应
    """
    stream = main_agent.stream({
        "messages": [
            {"role": "user", "content": query}
        ]
    })

    for chunk in stream:
        # node_name 是 "model" 或 "tools"，state 是对应的状态字典
        for node_name, state in chunk.items():
            if state is None or "messages" not in state:
                continue
            messages = state["messages"]
            if messages and isinstance(messages, list):
                last_messages = messages[-1]
                if node_name == "model":
                    if last_messages.tool_calls:
                        for tool_call in last_messages.tool_calls:
                            name = tool_call["name"]
                            args = tool_call["args"]
                            if name == "task":
                                print(f"[-agent-] 调用子智能体: {args.get('subagent_type')}")
                            else:
                                print(f"[-agent-] 调用工具{name}: {args}")
                        # for tool_call in last_messages.tool_calls:
                        #     if tool_call["name"] == "task":
                        #         print(f"[-agent-] 调用子智能体: {tool_call.args.subagent_type}")
                        #     else:
                        #         print(f"[-agent-] 调用工具{tool_call.name}: {tool_call.args}")
                    elif last_messages.content:
                        content = last_messages.content
                        print(f"[-agent-{node_name}] {content}")
                elif node_name == "tools":
                    name = last_messages.name
                    content = last_messages.content
                    print(f"[-agent调用工具-{name}] {content}")


# test_stream("请帮我查询北京的天气，并计算 123 + 456 的结果，然后将结果翻译成英文。")

# test_stream("1+1等于多少？")

async def test_stream_async(query):
    """
    测试主智能体的功能
    :param query: 用户输入的查询
    :return: 主智能体的响应
    """
    stream = main_agent.astream({
        "messages": [
            {"role": "user", "content": query}
        ]
    })

    async for chunk in stream:
        # node_name 是 "model" 或 "tools"，state 是对应的状态字典
        for node_name, state in chunk.items():
            if state is None or "messages" not in state:
                continue
            messages = state["messages"]
            if messages and isinstance(messages, list):
                last_messages = messages[-1]
                if node_name == "model":
                    if last_messages.tool_calls:
                        for tool_call in last_messages.tool_calls:
                            name = tool_call["name"]
                            args = tool_call["args"]
                            if name == "task":
                                print(f"[-agent-] 调用子智能体: {args.get('subagent_type')}")
                            else:
                                print(f"[-agent-] 调用工具{name}: {args}")
                        # for tool_call in last_messages.tool_calls:
                        #     if tool_call["name"] == "task":
                        #         print(f"[-agent-] 调用子智能体: {tool_call.args.subagent_type}")
                        #     else:
                        #         print(f"[-agent-] 调用工具{tool_call.name}: {tool_call.args}")
                    elif last_messages.content:
                        content = last_messages.content
                        print(f"[-agent-{node_name}] {content}")
                elif node_name == "tools":
                    name = last_messages.name
                    content = last_messages.content
                    print(f"[-agent调用工具-{name}] {content}")


if __name__ == '__main__':
    async def batch_run():
        task1 = test_stream_async("1+1等于多少？")
        task2 = test_stream_async("将我愿你天天开心翻译成英文。")
        await asyncio.gather(task1, task2)


    asyncio.run(
        batch_run()
    )

# =============================================================================
# Chunk 数据结构参考文档 (Python 对象视图)
# =============================================================================
# LangGraph 流式输出 (chunk) 的四种核心场景示例：
# 1. [场景 A：Agent 思考并决定调用工具]
#    {
#      "model": {
#        "messages": [
#          AIMessage(
#            content="",
#            tool_calls=[{
#              "name": "read_file_content",
#              "args": {"filename": "需求.docx"},
#              "id": "call_123"
#            }]
#          )
#        ]
#      }
#    }
# 2. [场景 B：工具执行完毕，返回结果]
#    {
#      "tools": {
#        "messages": [
#          ToolMessage(
#            content="[文件内容]...",
#            name="read_file_content",
#            tool_call_id="call_123"
#          )
#        ]
#      }
#    }
# 3. [场景 C：Agent 决定调用子 Agent (特殊工具 'task')]
#    {
#      "model": {
#        "messages": [
#          AIMessage(
#            content="",
#            tool_calls=[{
#              "name": "task",
#              "args": {
#                "subagent_type": "网络搜索助手",  # 目标子 Agent
#                "description": "查询2024政策"     # 下发的具体任务
#              },
#              "id": "call_456"
#            }]
#          )
#        ]
#      }
#    }
# 4. [场景 D：Agent 最终回复用户]
#    {
#      "model": {
#        "messages": [
#          AIMessage(
#            content="根据查询结果，2024年新政策如下...",
#            tool_calls=[]
#          )
#        ]
#      }
#    }
# =============================================================================
