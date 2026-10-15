# _*_ coding : utf-8 _*_
# @Time : 2026/10/15 21:11
# @Author : KarryLiu
# File : deep_agent_01
# @Project : DeepAgent
"""
案例1：体验DeepAgents创建，学习智能体的执行与结果解析
    步骤1：准本一个网络搜索工具
    步骤2：初始化DeepSeek对象
    步骤3：创建深度智能体
    步骤4：执行智能体并解析结果，分析其执行的结果
"""
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
from langchain.chat_models import init_chat_model

from langchain.tools import tool
from deepagents import create_deep_agent
from typing import Literal
from tavily import TavilyClient
import os

llm_name = os.getenv("LLM_DEEPSEEK")
tavily_key = os.getenv("TAVILY_API_KEY")

tavily_client = TavilyClient(api_key=tavily_key)


# 步骤1：准本一个网络搜索工具
def internet_search(
        query: str,
        max_results: int = 5,
        topic: Literal["news", "finance", "general"] = "general",
        include_raw_content: bool = False,
) -> dict:
    """
    通过Tavily API进行互联网搜索
    :param query: 查询关键词
    :param max_results: 最大返回结果数
    :param topic: 搜索主题
    :param include_raw_content: 是否包含原始内容
    :return: 搜索结果
    """
    print(f"开始搜索，参数为{query}, {max_results}, {topic}, {include_raw_content}")
    return tavily_client.search(
        query=query,
        max_results=max_results,
        topic=topic,
        include_raw_content=include_raw_content)


if not llm_name:
    raise ValueError("环境变量 LLM_DEEPSEEK 未配置")
if not tavily_key:
    raise ValueError("环境变量 TAVILY_API_KEY 未配置")

# 步骤2：初始化DeepSeek对象
llm = init_chat_model(
    model=llm_name,
    model_provider="openai"
)

# 步骤3：创建深度智能体
deep_agent = create_deep_agent(
    model=llm,
    tools=[internet_search],
    subagents=[],  # 配置子智能
    system_prompt="""
    你是一个高级研究员，专注于互联网搜索和信息分析。
    你的任务是根据用户的查询，使用提供的工具进行深入的网络搜索，并提供详细、准确的结果。
    请确保在回答中引用可靠来源，并对搜索结果进行分析和总结。
    """
)

result = deep_agent.invoke(
    {"messages": [{"role": "user", "content": "请帮我搜索关于人工智能在医疗领域的最新应用，并提供详细的分析和总结。"}]})

print(result["messages"][-1].content)



"""
{
  'messages': [
    HumanMessage(
        content='请帮我搜索关于人工智能在医疗领域的最新应用，并提供详细的分析和总结。',
        id='64f4f0ba-f96c-49e7-b2b1-d0566114e1bb'
    ),

    AIMessage(
        content='我将为您搜索关于人工智能在医疗领域的最新应用。让我先进行多角度的搜索。',
        response_metadata={
            'model_provider': 'openai',
            'model_name': 'deepseek-flash',
            'finish_reason': 'tool_calls',
            'token_usage': {...}
        },
        tool_calls=[
            {name: 'internet_search', args: {query: '人工智能在医疗领域的最新应用 2024 2025', max_results: 8, topic: 'general'}, id: 'call_00_...', type: 'tool_call'},
            {name: 'internet_search', args: {query: 'AI in healthcare latest applications 2025', max_results: 8, topic: 'general'}, id: 'call_01_...', type: 'tool_call'},
            {name: 'internet_search', args: {query: 'generative AI healthcare clinical applications breakthrough', max_results: 8, topic: 'general'}, id: 'call_02_...', type: 'tool_call'}
        ]
    ),

    ToolMessage(
        content='{"query": "...", "results": [ {url, title, content, score, id}, ... ]}',
        name='internet_search',
        tool_call_id='call_00_...'
    ),
    ToolMessage(
        content='{"query": "...", "results": [ {url, title, content, score, id}, ... ]}',
        name='internet_search',
        tool_call_id='call_01_...'
    ),
    ToolMessage(
        content='{"query": "...", "results": [ {url, title, content, score, id}, ... ]}',
        name='internet_search',
        tool_call_id='call_02_...'
    ),

    AIMessage(
        content='我已获得初步资料。让我进一步搜索几个关键细分领域以获得更深入的信息。',
        response_metadata={
            'model_provider': 'openai',
            'model_name': 'deepseek-flash',
            'finish_reason': 'tool_calls',
            'token_usage': {...}
        },
        tool_calls=[
            {name: 'internet_search', args: {query: 'AI medical imaging diagnosis FDA approved 2025 breakthrough', max_results: 6, topic: 'general'}, id: 'call_00_...', type: 'tool_call'},
            {name: 'internet_search', args: {query: 'AI drug discovery protein structure 2025 breakthrough', max_results: 6, topic: 'general'}, id: 'call_01_...', type: 'tool_call'},
            {name: 'internet_search', args: {query: '中国 医疗大模型 应用 2025 医院', max_results: 6, topic: 'general'}, id: 'call_02_...', type: 'tool_call'},
            {name: 'internet_search', args: {query: 'AI agent healthcare ambient scribe 2025 adoption', max_results: 6, topic: 'general'}, id: 'call_03_...', type: 'tool_call'}
        ]
    ),

    ToolMessage(content='{"query": "...", "results": [...]}', name='internet_search', tool_call_id='call_00_...'),
    ToolMessage(content='{"query": "...", "results": [...]}', name='internet_search', tool_call_id='call_01_...'),
    ToolMessage(content='{"query": "...", "results": [...]}', name='internet_search', tool_call_id='call_02_...'),
    ToolMessage(content='{"query": "...", "results": [...]}', name='internet_search', tool_call_id='call_03_...'),

    AIMessage(
        content='让我再补充搜索关于挑战、监管和投资方面的信息，以提供更全面的分析。',
        response_metadata={
            'model_provider': 'openai',
            'model_name': 'deepseek-flash',
            'finish_reason': 'tool_calls',
            'token_usage': {...}
        },
        tool_calls=[
            {name: 'internet_search', args: {query: 'AI healthcare challenges ethics regulation 2025 bias data privacy', max_results: 6, topic: 'general'}, id: 'call_00_...', type: 'tool_call'},
            {name: 'internet_search', args: {query: 'AI healthcare investment funding market size 2025', max_results: 6, topic: 'finance'}, id: 'call_01_...', type: 'tool_call'}
        ]
    ),

    ToolMessage(content='{"query": "...", "results": [...]}', name='internet_search', tool_call_id='call_00_...'),
    ToolMessage(content='{"query": "...", "results": [...]}', name='internet_search', tool_call_id='call_01_...'),

    AIMessage(
        content='# 人工智能在医疗领域的最新应用：深度分析与总结 ...（最终报告正文）',
        response_metadata={
            'model_provider': 'openai',
            'model_name': 'deepseek-flash',
            'finish_reason': 'stop',
            'token_usage': {...}
        },
        tool_calls=[]
    )
  ],
  'files': {}
}
"""