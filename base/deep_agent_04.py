# _*_ coding : utf-8 _*_
# @Time : 2026/10/19 15:44
# @Author : KarryLiu
# File : deep_agent_04
# @Project : DeepAgent
import asyncio
import os
from typing import TypedDict, Annotated

from deepagents import create_deep_agent, CompiledSubAgent
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
from langgraph.graph import START, END
from langgraph.graph import add_messages, StateGraph

from langchain.chat_models import init_chat_model

llm = init_chat_model(
    model=os.getenv("LLM_DEEPSEEK"),
    model_provider="openai"
)

# 定义一个LangGraph节点
class SubState(TypedDict):
    """
    定义的state必须包含 messages属性 【deepagents】，
    且messages属性必须是Annotated类型，追加add_messages
    """
    messages: Annotated[list, add_messages]
    new_message: dict[str, str]


def process_node(state: SubState) -> SubState:
    print("处理节点，当前状态为：", state)
    # 在这里可以对state进行处理，例如添加新的消息
    new_message = {"role": "assistant", "content": "这是处理节点后的新消息。"}
    state["messages"].append(new_message)
    print("处理节点完成，更新后的状态为：", state)
    return state


graph = StateGraph(SubState)
graph.add_node("process_node", process_node)

(graph.add_edge(START, "process_node")
 .add_edge("process_node", END))

app = graph.compile()

# 包装成DeepAgent认识的智能体

sub_agent = CompiledSubAgent(
    name="graph_sub_agent",
    description="这是一个基于LangGraph的子智能体，处理消息状态。",
    runnable=app
)

# 创建主智能体，包含一个子智能体
main_agent = create_deep_agent(
    model=llm,
    name="main_agent_with_graph",
    subagents=[sub_agent],
    system_prompt="这是一个主智能体，负责协调基于LangGraph的子智能体的调用，所有任务都需要经过子智能体处理。",
)

for msg in main_agent.stream({
    "messages": [
        {"role": "user", "content": "请处理这条消息。"}
    ]
}):
    for node_name, state in msg.items():
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
                elif last_messages.content:
                    content = last_messages.content
                    print(f"[-agent-{node_name}] {content}")
            elif node_name == "tools":
                name = last_messages.name
                content = last_messages.content
                print(f"[-agent调用工具-{name}] {content}")
