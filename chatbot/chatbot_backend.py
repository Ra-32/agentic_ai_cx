from langgraph.graph.state import LastValue
from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage,SystemMessage
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Literal
import os
load_dotenv()
CONFIG={
    'configurable':{
        'thread_id':'thread-1'
    }
}

groq_api_key=os.getenv("GROQ_API_KEY")


model=ChatGroq(model="llama-3.1-8b-instant",temperature=0.2,api_key=groq_api_key)
class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

graph=StateGraph(ChatState)
checkpointer=InMemorySaver()

def chat_node(state :ChatState):

    messages=state['messages']

    response=model.invoke(messages).content

    return {
        **state,
        'messages':[response]
    }

graph.add_node('chat_node',chat_node)

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

chatbot=graph.compile(checkpointer=checkpointer)


