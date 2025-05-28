import os
import contextlib
from llama_cpp import Llama

from Pipeline.philosophy_pipeline import PhilosophyPipeline
from Evaluator.metrics import compute_metrics, print_metrics

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

# Example queries and their true answers (Ground truth)
queries = [
    "Who is the first philosopher?",
    "What is existentialism?",
    # Add more queries as needed
]

# True labels or true answers for your queries
y_true = [
    "Thales of Miletus",
    "A philosophical theory emphasizing individual existence, freedom, and choice.",
    # Add corresponding true answers
]

y_pred = []

# Initialize LLaMA silently
with open(os.devnull, 'w') as fnull:
    with contextlib.redirect_stderr(fnull):
        llm = Llama(
            model_path=model_path,
            verbose=False
        )

# Generate predictions for each query
for query in queries:
    built_prompt = pipeline.build_prompt(query)
    response = llm(built_prompt, max_tokens=200)
    prediction = response['choices'][0]['text'].strip() # type: ignore
    y_pred.append(prediction)
    print(f"\n=== Answer to '{query}' ===")
    print(prediction)

# Compute and print evaluation metrics
metrics = compute_metrics(y_true, y_pred)
print_metrics(metrics)
