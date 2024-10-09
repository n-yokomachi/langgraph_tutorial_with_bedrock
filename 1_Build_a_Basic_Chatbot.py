from typing import Annotated
from langchain_aws import ChatBedrockConverse
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# StateクラスでNodeなどで利用するオブジェクトと、その更新方法をreducer関数として定義
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# chatbotがinvokeするLLMはChatBedrockConverseでConverse APIを実行するように変更
llm = ChatBedrockConverse(model="anthropic.claude-3-5-sonnet-20240620-v1:0")

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# chatbot関数をgraph_builder.add_node()でNodeとして追加
graph_builder.add_node("chatbot", chatbot)
# graph_builder.set_entry_point(), graph_builder.set_finish_point()でグラフの始点、終点をEdgeとして定義
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")

graph = graph_builder.compile()


def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


# このWhile文はこのグラフをチャットボットとして利用するための入力制御
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