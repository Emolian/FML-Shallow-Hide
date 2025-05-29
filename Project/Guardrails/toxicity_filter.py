class ToxicityChecker:
    def __init__(self, llm):
        self.llm = llm

    def is_toxic(self, text):
        prompt = (
            "You are a toxicity detection assistant. Analyze the following text and "
            "respond with only 'TOXIC' or 'SAFE'.\n\n"
            f"Text:\n{text}\n\nResponse:"
        )
        response = self.llm.generate(prompt).strip().upper()
        return response == "TOXIC"

