import os
import contextlib
from llama_cpp import Llama

from Pipeline.philosophy_pipeline import PhilosophyPipeline


# Get base dir where main.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build relative paths
csv_path = os.path.join(BASE_DIR, "Data", "philosophy_data.csv")
model_path = os.path.join(BASE_DIR, "Models", "Llama-3.2-3B-Instruct-Q4_K_M.gguf")

# Load and process the dataset
pipeline = PhilosophyPipeline(csv_path)
df = pipeline.load_and_clean_data()
pipeline.prepare_chunks_and_metadata(df)
pipeline.build_faiss_index()

# User query
user_query = "Who is the first philosopher?"
built_prompt = pipeline.build_prompt(user_query)

# Initialize LLaMA silently
with open(os.devnull, 'w') as fnull:
    with contextlib.redirect_stderr(fnull):
        llm = Llama(
            model_path=model_path,
            verbose=False
        )

# Generate response
response = llm(built_prompt, max_tokens=200)
print("\n=== Answer ===")
print(response['choices'][0]['text'].strip())
