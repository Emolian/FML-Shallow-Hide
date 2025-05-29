from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class HallucinationChecker:
    def __init__(self, threshold=0.15):
        self.threshold = threshold

    def is_hallucinated(self, context_chunks, generated_text):
        context_text = " ".join(context_chunks)
        vectorizer = TfidfVectorizer().fit([context_text, generated_text])
        vectors = vectorizer.transform([context_text, generated_text])
        sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        return sim < self.threshold, sim



def explain_hallucination_check(similarity_score, threshold):
    """
    Explains the hallucination likelihood based on the similarity score.

    Parameters:
        similarity_score (float): The actual similarity value.
        threshold (float): The configured cutoff for hallucination detection.

    Returns:
        str: Explanation string.
    """
    if similarity_score < threshold:
        return f"Similarity score {similarity_score:.2f} < threshold {threshold} → possible hallucination."
    return f"Similarity score {similarity_score:.2f} ≥ threshold {threshold} → contextually grounded."