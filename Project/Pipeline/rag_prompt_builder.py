def build_prompt(user_query, context_chunks, system_instruction=None):
    """
    Builds a prompt by prepending relevant context and optionally a system-level instruction.

    Parameters:
        user_query (str): The question or instruction from the user.
        context_chunks (list of str): Retrieved relevant text chunks.
        system_instruction (str, optional): Optional guidance for the LLM (e.g. "You are a helpful philosophy tutor").

    Returns:
        str: The complete prompt for the LLM.
    """
    context = "\n".join(context_chunks)

    if system_instruction:
        return f"""System: {system_instruction}

Context:
{context}

Question: {user_query}
Answer:"""

    return f"""Context:
{context}

Question: {user_query}
Answer:"""
