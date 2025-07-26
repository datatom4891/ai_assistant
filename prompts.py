from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate


#Tavily Prompt
tavily_prompt_template ="""
                        Answer the question below only using the provided context.
                        Be concise answering with 1-3 sentences only.

                        Context: {context}

                        Question: {question}
                        """

tavily_prompt = ChatPromptTemplate.from_template(tavily_prompt_template)