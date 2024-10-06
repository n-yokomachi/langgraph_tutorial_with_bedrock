# 最初に基本的なチャットボットを構築します。
# このチャットボットは、ユーザーの入力を受け取り、そのまま返すだけのシンプルなものです。

from typing import Annotated

from langchain_aws import ChatBedrockConverse
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages



class State(TypedDict):
    # メッセージの型は "list" です。アノテーションの `add_messages` 関数は
    # この状態キーをどのように更新するべきかを定義します
    # （この場合、メッセージを上書きするのではなく、リストに追加します）
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

llm = ChatBedrockConverse(model="anthropic.claude-3-5-sonnet-20240620-v1:0")


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# 最初の引数はユニークなノード名です
# 2番目の引数は、ノードが使用されるたびに呼び出される関数またはオブジェクトです
graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")
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
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break