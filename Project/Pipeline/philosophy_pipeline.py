import os
import json
import yaml
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class PhilosophyPipeline:
    def __init__(self, csv_path, config_path="Config/rag_config.yaml"):
        self.csv_path = csv_path
        self.config = self._load_config(config_path)


        self.chunk_size = self.config["retrieval"]["chunk_size"]
        self.top_k = self.config["retrieval"]["top_k"]
        self.max_context_tokens = self.config["retrieval"]["max_context_tokens"]
        self.index_path = self.config["retrieval"]["faiss_index_path"]
        self.meta_path = self.config["retrieval"]["metadata_path"]

        model_name = self.config["retrieval"]["embedding_model"]
        self.embedding_model = SentenceTransformer(model_name)

        self.chunks = []
        self.metadata = []
        self.index = None

    def load_and_clean_data(self):
        import pandas as pd
        df = pd.read_csv(self.csv_path)
        df = df.dropna(subset=["sentence_str"])
        df['sentence'] = df['sentence_str'].str.strip()
        df['author'] = df['author'].str.strip().str.lower()
        df['school'] = df['school'].str.strip().str.lower()
        return df

    def _load_config(self, path):
        import yaml
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        if not os.path.isabs(path):
            path = os.path.normpath(os.path.join(base_dir, path))

        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if "retrieval" in config:
            retrieval_cfg = config["retrieval"]
            retrieval_cfg["faiss_index_path"] = os.path.normpath(
                os.path.join(base_dir, retrieval_cfg["faiss_index_path"]))
            retrieval_cfg["metadata_path"] = os.path.normpath(os.path.join(base_dir, retrieval_cfg["metadata_path"]))

        return config

    def retrieve_context(self, query):
        query_vec = self.embedding_model.encode([query])
        distances, indices = self.index.search(np.array(query_vec), self.top_k)

        valid_indices = [i for i in indices[0] if i < len(self.chunks)]

        return [self.chunks[i] for i in valid_indices], [self.metadata[i] for i in valid_indices]

    def build_prompt(self, user_query, examples=None, role=None, instruction=None):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config["retrieval"]["embedding_model"])

        # Retrieve relevant context
        context_chunks, _ = self.retrieve_context(user_query)

        # Limit context tokens
        total_tokens = 0
        selected_chunks = []
        for chunk in context_chunks:
            chunk_tokens = len(tokenizer.encode(chunk, add_special_tokens=False))
            if total_tokens + chunk_tokens > self.max_context_tokens:
                break
            selected_chunks.append(chunk)
            total_tokens += chunk_tokens

        context = "\n".join([f"{i + 1}. {c}" for i, c in enumerate(selected_chunks)])

        # Default role and instruction
        role = role or "You are a knowledgeable philosophy assistant. Your answers must be faithful to the provided context."
        instruction = instruction or (
            "Answer the user's question using ONLY the provided context. "
            "If the context does not contain enough information, say exactly: "
            "'The context does not provide a direct answer.'\n\n"
            "Do not make up information, even if the answer seems obvious. "
            "Do not repeat the question. Be concise, accurate, and grounded."
        )

        # Few-shot examples
        example_section = ""
        if examples:
            formatted = [f"Q: {q}\nA: {a}" for q, a in examples]
            example_section = "### Few-shot Examples\n" + "\n\n".join(formatted) + "\n\n"

        # Final prompt
        prompt = f"""### Role
    {role}

    ### Instruction
    {instruction}

    {example_section}### Context
    {context}

    ### Question
    {user_query}

    ### Answer"""

        return prompt.strip()

    def save_index_and_metadata(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

    def load_index_and_metadata(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def chunk_sentences(self, group):
        chunks = []
        sentences = group['sentence'].tolist()
        for i in range(0, len(sentences), self.chunk_size):
            chunk = " ".join(sentences[i:i + self.chunk_size])
            chunks.append(chunk)
        return chunks

    def prepare_chunks_and_metadata(self, df):
        grouped = df.groupby(['title', 'author'])
        for (title, author), group in grouped:
            chs = self.chunk_sentences(group)
            for ch in chs:
                self.chunks.append(ch)
                self.metadata.append({
                    'title': title,
                    'author': author,
                    'school': group['school'].iloc[0]
                })

    def build_faiss_index(self):
        embeddings = self.embedding_model.encode(self.chunks, show_progress_bar=True)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings))
        self.save_index_and_metadata()








