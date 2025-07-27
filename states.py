from langgraph.graph import MessagesState
from typing import List

class State(MessagesState):
    
    # add memories that will be retrieved based on the conversation context
    recall_memories: List[str]