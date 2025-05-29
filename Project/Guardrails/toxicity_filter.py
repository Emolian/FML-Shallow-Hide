import re
import csv

def load_toxic_keywords(path="Config/bad_words.csv"):
    """
    Load toxic keywords from a CSV file with optional categories.

    CSV Format:
    keyword,category

    Returns:
        List[str]: Toxic keywords to scan for.
    """
    keywords = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "keyword" in row:
                keywords.append(row["keyword"].strip().lower())
    return keywords


def is_toxic(text, toxic_keywords):
    """
    Checks if the given text contains any toxic language based on the keyword list.

    Parameters:
        text (str): Text to evaluate.
        toxic_keywords (List[str]): List of keywords to scan for.

    Returns:
        bool: True if toxic content is detected, False otherwise.
    """
    text_lower = text.lower()
    for keyword in toxic_keywords:
        if re.search(rf"\b{re.escape(keyword)}\b", text_lower):
            return True
    return False


def highlight_toxic_words(text, toxic_keywords):
    """
    Highlights toxic words found in the text with asterisks for debugging.

    Parameters:
        text (str): Input text.
        toxic_keywords (List[str]): List of keywords to highlight.

    Returns:
        str: Modified string with highlighted toxic terms.
    """
    result = text
    for keyword in toxic_keywords:
        pattern = re.compile(rf"\b({re.escape(keyword)})\b", re.IGNORECASE)
        result = pattern.sub(r"*\1*", result)
    return result
