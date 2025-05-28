import os
import contextlib
from llama_cpp import Llama
from Pipeline.philosophy_pipeline import PhilosophyPipeline

# Load and process the dataset
pipeline = PhilosophyPipeline("C:\\Users\\Lenovo\\PycharmProjects\\FMLProject\\Project\\Data\\philosophy_data.csv")
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
            model_path="C:\\Users\\Lenovo\\models\\Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            verbose=False
        )

# Generate response
response = llm(built_prompt, max_tokens=200)
print("\n=== Answer ===")
print(response['choices'][0]['text'].strip())