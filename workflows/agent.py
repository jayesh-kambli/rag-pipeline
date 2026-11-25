from langchain.agents import create_agent
from workflows.tools import calculator, search_ipl_knowledge, number_difference_tool
from langchain_openai import ChatOpenAI
import os

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-6-luna"),
    api_key=os.getenv("OPEN_AI_API_KEY")
)

SYSTEM_PROMPT = """You are an IPL assistant.
For any question about the IPL, ALWAYS call the search_ipl_knowledge tool first
and answer ONLY from the text it returns. Do not use outside knowledge.
If the answer is not in the returned text, say that the document does not contain it.
Keep answers short and clear. Use the calculator tools for any arithmetic."""


def create_ipl_agent():
    tools = [number_difference_tool, calculator, search_ipl_knowledge]
    agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)
    return agent
