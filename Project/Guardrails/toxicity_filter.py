import re

# Simple list of toxic keywords (can be expanded or replaced by a model)
TOXIC_KEYWORDS = [
    # General hate and slur indicators (non-exhaustive, redacted)
    "n****", "k***", "f****t", "tr****", "c**n", "r****d", "b****", "sl*t", "wh*re",

    # Masked violent expressions
    "kill all", "gas the", "hang the", "lynch", "shoot up", "burn them",

    # Extremist/terrorist phrases
    "white power", "heil hitler", "kkk", "zionist pig", "jihadist scum", "Free Palestine" #apparently the last one is as horrible as HH according to the Israeli prime minister

    # Anti-LGBTQ+ and misogyny
    "it's adam and eve", "man up", "feminazi", "women are inferior", "transgenders are sick",

    # Encouraging self-harm
    "kill yourself", "cut yourself", "no one loves you", "you're worthless", "kys", "Low Tier God", "LTG", "rope plus chair ikea combo", "Kazuha E out of a window","navia e in your head", "Raiden Q in reverse"

    # Philosophically dangerous without context
    "slavery is good", "the holocaust didn’t happen", "eugenics works", "it never happened, but they deserved it"

    # Toxic rhetorical framing
    "why do women lie", "are black people less intelligent", "is islam evil",

    # Masked insults
    "you’re too dumb", "worthless idiot", "stupid piece of"

    # Optionally: misspellings or leetspeak variations
    "n1gger", "f@g", "b1tch"
]


def is_toxic(text):
    """
    Checks if the given text contains any toxic language based on a keyword list.

    Parameters:
        text (str): The generated text to be evaluated.

    Returns:
        bool: True if toxic content is detected, False otherwise.
    """
    text_lower = text.lower()
    for keyword in TOXIC_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", text_lower):
            return True
    return False

def highlight_toxic_words(text):
    """
    Highlights toxic words found in the text for logging/debugging.

    Parameters:
        text (str): The input text.

    Returns:
        str: The text with toxic words surrounded by asterisks.
    """
    text_lower = text.lower()
    result = text
    for keyword in TOXIC_KEYWORDS:
        if keyword in text_lower:
            pattern = re.compile(rf"\b({re.escape(keyword)})\b", re.IGNORECASE)
            result = pattern.sub(r"*\1*", result)
    return result
