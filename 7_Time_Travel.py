from typing import Annotated
from langchain_aws import ChatBedrockConverse
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]
    # ユーザーNodeを呼び出すかのフラグ
    ask_human: bool
   
 
class RequestAssistance(BaseModel):
    """
    専門家に会話をエスカレーションします。直接支援できない場合や、ユーザーがあなたの権限を超えるサポートを必要とする場合にこれを使用してください。
    この機能を使用するには、専門家が適切なガイダンスを提供できるように、ユーザーの'request'（要求）を伝えてください。
    """
    request: str
    

# toolを定義
tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatBedrockConverse(model="anthropic.claude-3-5-sonnet-20240620-v1:0")
# LLMに使用できるtoolとPydanticモデルをバインド可能
llm_with_tools = llm.bind_tools(tools + [RequestAssistance])


def chatbot(state: State):
    # LLMをinvoke
    response = llm_with_tools.invoke(state["messages"])
    
    # LLMのレスポンスがtool useかつ
    ask_human = False
    if (
        response.tool_calls
        and response.tool_calls[0]["name"] == RequestAssistance.__name__
    ):
        ask_human = True
        
    return {"messages": [response], "ask_human": ask_human}
    

# グラフを定義
graph_builder = StateGraph(State)

# chatbot, toolsノードをグラフに定義
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=[tool]))



def create_response(response: str, ai_message: AIMessage):
    return ToolMessage(
        content=response,
        tool_call_id=ai_message.tool_calls[0]["id"])


def human_node(state: State):
    new_message = []
    if not isinstance(state["messages"][-1], ToolMessage):
        # 中断中にユーザーがStateを更新しない場合、ToolMessageを追加してLLMを続行させる
        new_message.append(
            create_response("No response from human.", state["messages"[-1]])
        )
    return{
        # 新しいメッセージを追加
        "messages": new_message,
        # フラグを解除
        "ask_human": False
    }

graph_builder.add_node("human", human_node)


def select_next_node(state: State):
    if state["ask_human"]:
        return "human"
    # 
    return tools_condition(state)


graph_builder.add_conditional_edges(
    "chatbot",
    select_next_node,
    {"human": "human", "tools": "tools", END: END},
)



graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("human", "chatbot")
graph_builder.set_entry_point("chatbot")
memory = MemorySaver()
graph = graph_builder.compile(
    checkpointer=memory, 
    interrupt_before=["human"]
)



config = {"configurable": {"thread_id": "1"}}
def stream_graph_updates(user_input: str):
    events = graph.stream({"messages": [("user", user_input)]}, config, stream_mode="values")
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()
            
            
def stream_graph_updates_by_none():
    events = graph.stream(None, config, stream_mode="values")
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()


def update_graph_state():
    snapshot = graph.get_state(config)
    ai_message = snapshot.values["messages"][-1]
    human_response = (
        "私たち専門家がお手伝いします！エージェントを構築するにはLangGraphをチェックすることをお勧めします。"
        "単純な自律エージェントよりもはるかに信頼性が高く、拡張性があります。"
    )
    tool_message = create_response(human_response, ai_message)
    graph.update_state(config, {"messages": [tool_message]})
    print(graph.get_state(config).values["messages"])


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        elif user_input == "":
            stream_graph_updates_by_none()
        elif user_input == "update":
            update_graph_state()
        else:
            stream_graph_updates(user_input)
    except:
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break