import ast
import operator

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool 

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(embedding_function=embeddings, collection_name="ipl_langchain", persist_directory="croma_langchain")
retriever = vector_store.as_retriever(search_kwargs={"k": 10})

@tool
def number_difference_tool(
    first_value: float,
    second_value: float
) -> float:
    """
    Use this tool ONLY when the user asks
    to perform any difference or subtraction
    between two numerical values.
    """

    print("Running Number Difference tool................")
    return first_value - second_value


@tool
def calculator(
    expression: str
) -> str:
    """
    Perform mathematical calculations.
    Input should be a valid mathematical expression string like '2+2' or '5*(3+2)'.
    """
    print("Calculating........................")
    try:
        return str(_safe_eval(ast.parse(expression, mode="eval").body))
    except Exception:
        return "Invalid expression. Use only numbers and + - * / ** % ( )."


# Only plain arithmetic is allowed, so the app is safe to put online
_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}

def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


@tool
def search_ipl_knowledge(
    question: str
) -> str:
    """
    Use this tool when the user is asking about
    IPL facts, statistics, players, teams, matches,
    seasons, records, or other information that can
    be answered from the TEXT of the IPL document.
    """

    print("Running Rag Tool...")

    docs = retriever.invoke(question)

    # Label every chunk with where it came from, so the app can show sources
    blocks = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "?")
        blocks.append(f"[{source} | chunk {chunk_id}]\n{doc.page_content}")

    return SOURCE_SEPARATOR.join(blocks)


SOURCE_SEPARATOR = "\n\n-----\n\n"