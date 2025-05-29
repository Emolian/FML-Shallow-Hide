import os
from Evaluator.metrics import compute_metrics, print_metrics
from Runner.llama_wrapper import LlamaWrapper
from Pipeline.philosophy_pipeline import PhilosophyPipeline

# Base directory (Project/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Paths
csv_path = os.path.join(BASE_DIR, "Data", "philosophy_data.csv")
config_path_model = os.path.join(BASE_DIR, "Config", "model_setting.yaml")
config_path_rag = os.path.join(BASE_DIR, "Config", "rag_config.yaml")

# Initialize pipeline and model
pipeline = PhilosophyPipeline(csv_path, config_path=config_path_rag)
llm = LlamaWrapper(config_path=config_path_model)

# Check if FAISS index and metadata already exist
if os.path.exists(pipeline.index_path) and os.path.exists(pipeline.meta_path):
    print("🔄 Loading FAISS index and metadata from disk...")
    pipeline.load_index_and_metadata()
else:
    print("⚙️ Building FAISS index and metadata from scratch...")
    df = pipeline.load_and_clean_data()
    pipeline.prepare_chunks_and_metadata(df)
    pipeline.build_faiss_index()
    pipeline.save_index_and_metadata()

# Example queries
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

# Ground-truth answers
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

example_pairs = [
    ("What is Plato's theory of forms?", "Plato believed in a world of ideal forms, which physical objects imperfectly reflect."),
    ("What does Aristotle mean by 'final cause'?", "It refers to the purpose or end for which something exists."),
    ("What is the meaning of 'Cogito, ergo sum'?", "It means 'I think, therefore I am' — Descartes' proof of existence.")
]

# Run LLM over queries
y_pred = []
for query in queries:
    example_pairs = [
        ("What is Plato's theory of forms?",
         "Plato believed in a world of ideal forms, which physical objects imperfectly reflect."),
        ("What does Aristotle mean by 'final cause'?", "It refers to the purpose or end for which something exists."),
        ("What is the meaning of 'Cogito, ergo sum'?",
         "It means 'I think, therefore I am' — Descartes' proof of existence.")
    ]
    prompt = pipeline.build_prompt(query, examples=example_pairs)
    answer = llm.generate(prompt)
    y_pred.append(answer)
    print(f"\n=== Answer to '{query}' ===")
    print(answer)

# Evaluate results
metrics = compute_metrics(y_true, y_pred)
print_metrics(metrics)
