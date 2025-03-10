import asyncio
from typing import Sequence
from autogen_core import CancellationToken
from autogen_agentchat.agents import AssistantAgent, SocietyOfMindAgent, UserProxyAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination, HandoffTermination
from autogen_agentchat.messages import AgentEvent, ChatMessage, TextMessage, ToolCallExecutionEvent
from autogen_agentchat.teams import SelectorGroupChat, RoundRobinGroupChat, Swarm
from autogen_agentchat.ui import Console
from autogen_agentchat.base import Handoff
from autogen_ext.models.openai import OpenAIChatCompletionClient
from backend.constant import MODEL, OPENAI_API_KEY, OPENAI_BASE_URL
from backend.tools import vector_retrieval_from_knowledge_base, sendScheme2RobotWorkstation, sendScheme2MobileRobot, get_latest_exp_log, scheme_convert_to_json, upload_to_s3


async def _single_agent_answer_with_rag(user_query:str, model: str = MODEL):
    model_client = OpenAIChatCompletionClient(
        model=model,
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
        model_info={
            "vision": True,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
        },
    )
    try:
        assistant = AssistantAgent(
            name="assistant",
            system_message="""You are a helpful assistant. You can call tools to help user.""",
            model_client=model_client,
            tools=[vector_retrieval_from_knowledge_base],
            reflect_on_tool_use=True, # Set to True to have the model reflect on the tool use, set to False to return the tool call result directly.
        )

        response = await assistant.on_messages([TextMessage(content=user_query, source="user")], CancellationToken())
        return response.chat_message.content
    except:
        return "Sorry, I am not able to answer your question."
    # print("Assistant:", response.chat_message.content)


async def _single_agent_answer_with_rag_cot(user_query:str, model: str = MODEL):
    model_client = OpenAIChatCompletionClient(
        model=model,
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
        model_info={
            "vision": True,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
        },
    )

    assistant = AssistantAgent(
        name="assistant",
        system_message="""You are a helpful assistant. You can call tools to help user. Using chain of thought (CoT) when answering questions.""",
        model_client=model_client,
        tools=[vector_retrieval_from_knowledge_base],
        reflect_on_tool_use=True, # Set to True to have the model reflect on the tool use, set to False to return the tool call result directly.
    )

    response = await assistant.on_messages([TextMessage(content=user_query + "\nLet's think step by step:", source="user")], CancellationToken())
    return response.chat_message.content
    # print("Assistant:", response.chat_message.content)


async def main(model: str = MODEL):
    model_client = OpenAIChatCompletionClient(
        model=model,
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
        model_info={
            "vision": True,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
        },
    )

    assistant = AssistantAgent(
        name="assistant",
        system_message="""You are a helpful assistant. You can call tools to help user.""",
        model_client=model_client,
        tools=[vector_retrieval_from_knowledge_base],
        reflect_on_tool_use=True, # Set to True to have the model reflect on the tool use, set to False to return the tool call result directly.
    )

    while True:
        user_input = input("User: ")
        if user_input == "exit":
            break
        response = await assistant.on_messages([TextMessage(content=user_input, source="user")], CancellationToken())
        print("Assistant:", response.chat_message.content)


if __name__ == "__main__":
    # asyncio.run(main())

    # answer = asyncio.run(_single_agent_answer_with_rag("how to synthesis CsPbBr3 nanocubes at room temperature?", model="gpt-4o"))
    # answer = single_agent_answer_with_rag("how to synthesis CsPbBr3 nanocubes at room temperature?", model="gpt-4o")
    # print()
    pass
