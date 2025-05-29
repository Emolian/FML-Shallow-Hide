from sentence_transformers import util
from sentence_transformers import SentenceTransformer

class HallucinationChecker:
    def __init__(self, threshold=0.15, model_name='all-MiniLM-L6-v2'):
        self.threshold = threshold
        self.model = SentenceTransformer(model_name)

    def is_hallucinated(self, context_chunks, generated_text):
        context_text = " ".join(context_chunks)

        # Ensure both inputs are treated as lists to get 2D embeddings
        context_embedding = self.model.encode([context_text], convert_to_tensor=True)
        gen_embedding = self.model.encode([generated_text], convert_to_tensor=True)

        # Compute cosine similarity
        sim_tensor = util.cos_sim(context_embedding, gen_embedding)
        sim = float(sim_tensor[0][0])  # extract scalar from 1x1 tensor

        return sim < self.threshold, sim

    def explain(self, sim):
        if sim < self.threshold:
            return f"⚠️ Similarity {sim:.3f} < threshold {self.threshold} → likely hallucinated."
        return f"✅ Similarity {sim:.3f} ≥ threshold {self.threshold} → grounded in context."
