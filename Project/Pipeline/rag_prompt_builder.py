def build_prompt(user_query, context_chunks, system_instruction=None):
    """
    Builds a structured prompt for the LLM using retrieved context and an optional system instruction.

    Parameters:
        user_query (str): The user's question.
        context_chunks (list of str): Chunks retrieved from vector search.
        system_instruction (str, optional): Instruction for LLM behavior.

    Returns:
        str: The formatted prompt to be passed to the LLM.
    """
    context = "\n".join(context_chunks)

    if system_instruction is None:
        system_instruction = (
            "You are the world's best philosophy assistant and expert on the history of philosophy. "
            "Answer the user's question using only the provided context. Do not make up information. "
            "If the answer is not present in the context, say: 'The context does not provide a direct answer.'"
        )

    prompt = f"""
### System:
{system_instruction}

### Context:
{context}

### Question:
{user_query}

### Answer:
"""

    return prompt.strip()

