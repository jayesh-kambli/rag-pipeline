from langchain.agents import create_agent
from workflows.tools import calculator, search_ipl_knowledge, number_difference_tool
from langchain_openai import ChatOpenAI
import os

llm = ChatOpenAI(
    model="gpt-6-luna",
    api_key=os.getenv("OPEN_AI_API_KEY")
)

def create_ipl_agent():
    tools = [number_difference_tool, calculator, search_ipl_knowledge]
    agent = create_agent(model=llm, tools=tools)
    return agent
