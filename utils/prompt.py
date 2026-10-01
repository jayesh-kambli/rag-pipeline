def ipl_prompt(context, question):
    return f"""
    
    You are an IPL data analyst.
    Your task is to answer the user's question using only the IPL information provided below.
    IPL INFORMATION: 
    `{context}`

    USER QUESTION:
    `{question}`

    CONSTRAINTS:
    - Use only the provided information.
    - Do not invent facts or statistic
    - Do not make claims that cannot be orted by the context.
    - If the information required to answer the question is unavailable, clearly state that it is unavailable.

    OUTPUT FORMAT:
    Answer:
    <direct answer>

    Supporting Information:
    <relevant facts or statistics from the provided context>
    
    Limitations:
    <mention any limitation if the provided information is insufficient>
    """