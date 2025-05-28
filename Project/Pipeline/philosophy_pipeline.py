import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class PhilosophyPipeline:
    def __init__(self, csv_path, embedding_model='all-MiniLM-L6-v2', chunk_size=5):
        self.csv_path = csv_path
        self.chunk_size = chunk_size
        self.embedding_model = SentenceTransformer(embedding_model)
        self.chunks = []
        self.metadata = []
        self.index = None

    def load_and_clean_data(self):
        df = pd.read_csv(self.csv_path)
        df = df.dropna(subset=["sentence_str"])
        df['sentence'] = df['sentence_str'].str.strip()
        df['author'] = df['author'].str.strip().str.lower()
        df['school'] = df['school'].str.strip().str.lower()
        return df

    def chunk_sentences(self, group):
        chunks = []
        sentences = group['sentence'].tolist()
        for i in range(0, len(sentences), self.chunk_size):
            chunk = " ".join(sentences[i:i+self.chunk_size])
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

    def retrieve_context(self, query, top_k=3):
        query_vec = self.embedding_model.encode([query])
        distances, indices = self.index.search(np.array(query_vec), top_k)
        return [self.chunks[i] for i in indices[0]], [self.metadata[i] for i in indices[0]]

    def build_prompt(self, user_query):
        context_chunks, _ = self.retrieve_context(user_query)
        context = "\n".join(context_chunks)
        return f"""Context: \n\n{context}\n\nQuestion: \n\n{user_query}\n\nAnswer:\n\n"""



