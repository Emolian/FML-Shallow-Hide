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
    "What did Plato mean by the world of forms (ideas)?",
    "How does Aristotle define causality?",
    "What does Descartes mean by 'I think, therefore I am'?",
    "What is the Stoic view of virtue and happiness?",
    "What is Nietzsche’s concept of the will to power?",
    "How does Sartre define human freedom?",
    "What is 'Dasein' in Heidegger's philosophy?",
    "What critique does Kant offer against empiricism?",
    "What is the dialectical method according to Hegel?",
    "What are the differences between utilitarianism and deontology?"
]

y_true = [
    "Plato's world of forms refers to a realm of perfect, immutable concepts or ideals that exist independently of the physical world. Physical objects are merely imperfect copies of these forms.",
    "Aristotle defined four causes: material, formal, efficient, and final. He believed understanding something requires knowing all four causes, especially the final cause, or its purpose.",
    "Descartes’ statement 'Cogito, ergo sum' asserts that the act of thinking proves the existence of the self as a thinking being.",
    "Stoics believed that virtue is the only true good and that happiness (eudaimonia) is achieved by living in accordance with reason and nature.",
    "Nietzsche’s will to power is the fundamental driving force in humans, an instinct to grow, assert control, and overcome obstacles.",
    "Sartre defined freedom as the core of human existence; individuals are condemned to be free and bear full responsibility for their choices.",
    "In Heidegger’s philosophy, Dasein refers to the human being as a being that is aware of and questions its own existence.",
    "Kant argued that while knowledge begins with experience, not all knowledge arises from experience; the mind structures experience through a priori concepts.",
    "Hegel’s dialectical method involves a triadic process: thesis, antithesis, and synthesis, through which reality and ideas evolve.",
    "Utilitarianism judges actions by their consequences and seeks the greatest good for the greatest number, while deontology judges actions by whether they follow moral rules or duties."
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
    response = llm(built_prompt, max_tokens=256)
    prediction = response['choices'][0]['text'].strip() # type: ignore
    y_pred.append(prediction)
    print(f"\n=== Answer to '{query}' ===")
    print(prediction)

# Compute and print evaluation metrics
metrics = compute_metrics(y_true, y_pred)
print_metrics(metrics)

