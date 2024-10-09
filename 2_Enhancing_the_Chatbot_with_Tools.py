from typing import Annotated
from langchain_aws import ChatBedrockConverse
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Tavily Search APIのAPIキーは.envファイルに記述し読み込む
from dotenv import load_dotenv
load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# Web検索を行うAPIとしてTavily Search APIを使うように設定
tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatBedrockConverse(model="anthropic.claude-3-5-sonnet-20240620-v1:0")
# bind_tools()でLLMが使えるToolの紐づけ（tool use）を定義
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

# add_node()でtoolをNodeに追加
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

# 事前定義された関数tools_condition()を条件付きエッジとして定義
# （tools_condition()はLLM実行後のメッセージが「tool_calls」だった場合、toolsノードを呼び出すように定義されている）
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile()


def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break