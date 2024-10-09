from typing import Annotated
from langchain_aws import ChatBedrockConverse
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()

from dotenv import load_dotenv
load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatBedrockConverse(model="anthropic.claude-3-5-sonnet-20240620-v1:0")
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
# interrupt_beforeを指定し、toolsの使用前に中断を入れるように設定
graph = graph_builder.compile(
    checkpointer=memory, 
    interrupt_before=["tools"]
)


# 会話で使用するスレッドIDを指定
config = {"configurable": {"thread_id": "1"}}
# ユーザー入力がある場合
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}, config):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)
# ユーザー入力がない場合はNoneを渡してstreaming
# Noneが渡された場合、中断部分から再開する
def stream_graph_updates_by_none():
    for event in graph.stream(None, config):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        elif user_input == "":
            stream_graph_updates_by_none()
        else:
            stream_graph_updates(user_input)
    except:
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break