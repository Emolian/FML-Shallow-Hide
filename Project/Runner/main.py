import os
from Evaluator.metrics import compute_metrics, print_metrics
from Runner.llama_wrapper import LlamaWrapper
from Pipeline.philosophy_pipeline import PhilosophyPipeline
from Guardrails.toxicity_filter import ToxicityChecker
from Guardrails.hallucination_checker import HallucinationChecker
# Base directory (Project/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Paths
csv_path = os.path.join(BASE_DIR, "Data", "philosophy_data.csv")
config_path_model = os.path.join(BASE_DIR, "Config", "model_setting.yaml")
config_path_rag = os.path.join(BASE_DIR, "Config", "rag_config.yaml")

# Initialize pipeline and model
pipeline = PhilosophyPipeline(csv_path, config_path=config_path_rag)
llm = LlamaWrapper(config_path=config_path_model)
hallucination_checker = HallucinationChecker(threshold=0.15)
toxicity_checker = ToxicityChecker(llm=llm)

# Check if FAISS index and metadata already exist
if os.path.exists(pipeline.index_path) and os.path.exists(pipeline.meta_path):
    print("\U0001F501 Loading FAISS index and metadata from disk...")
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
    "Plato believed that the physical world is a shadow of a higher world of Forms—eternal, perfect ideals.",
    "Aristotle identified four causes—material, formal, efficient, and final—to explain why things exist or change.",
    "Descartes’ phrase 'Cogito, ergo sum' means that thinking is the proof of one's own existence.",
    "For Stoics, virtue is the only true good, and happiness comes from living according to reason and nature.",
    "Nietzsche’s 'will to power' is a fundamental drive behind human ambition, creativity, and overcoming challenges.",
    "Sartre argued that human freedom means we are condemned to choose, with no predefined essence or nature.",
    "'Dasein' is Heidegger's term for human existence, defined by being aware of and questioning one’s own being.",
    "Kant critiqued empiricism by asserting that knowledge arises from both sensory input and rational categories.",
    "Hegel’s dialectic method involves thesis, antithesis, and synthesis to explain the evolution of ideas and reality.",
    "Utilitarianism judges actions by consequences, aiming at happiness; deontology judges by adherence to moral duty."
]


example_pairs = [
    ("What is Plato's theory of forms?",
     "Plato believed in a world of ideal forms, which physical objects imperfectly reflect."),
    ("What does Aristotle mean by 'final cause'?", "It refers to the purpose or end for which something exists."),
    ("What is the meaning of 'Cogito, ergo sum'?",
     "It means 'I think, therefore I am' — Descartes' proof of existence.")
]

y_pred = []
for query in queries:
    for attempt in range(3):  # Retry up to 3 times
        prompt = pipeline.build_prompt(query, examples=example_pairs)
        answer = llm.generate(prompt)

        context_chunks, _ = pipeline.retrieve_context(query)
        is_hallucinated, similarity = hallucination_checker.is_hallucinated(context_chunks, answer)
        is_toxic = toxicity_checker.is_toxic(answer)

        if is_hallucinated:
            print(f"\u26a0\ufe0f Hallucination detected (sim={similarity:.3f}), regenerating...")
        elif is_toxic:
            print("\u26a0\ufe0f Toxic response detected, regenerating...")
        else:
            break  # Valid output

    y_pred.append(answer)
    print(f"\n=== Answer to '{query}' ===")
    print(answer)
    print(f"Hallucination: {'YES' if is_hallucinated else 'NO'} (sim={similarity:.3f})")
    print(f"Toxic: {'YES' if is_toxic else 'NO'}")

# Evaluate
metrics = compute_metrics(y_true, y_pred)
print_metrics(metrics)

# === Interactive Chatbot Mode ===
print("\n🤖 Entering interactive chatbot mode. Ask philosophical questions (type 'exit' to quit):\n")

while True:
    user_input = input("🧠 You: ").strip()
    if user_input.lower() in {"exit", "quit"}:
        print("👋 Goodbye!")
        break

    for attempt in range(3):
        prompt = pipeline.build_prompt(user_input, examples=example_pairs)
        answer = llm.generate(prompt)

        context_chunks, _ = pipeline.retrieve_context(user_input)
        is_hallucinated, similarity = hallucination_checker.is_hallucinated(context_chunks, answer)
        is_toxic = toxicity_checker.is_toxic(answer)

        if is_hallucinated:
            print(f"\u26a0\ufe0f Hallucination detected (sim={similarity:.3f}), regenerating...")
        elif is_toxic:
            print("\u26a0\ufe0f Toxic response detected, regenerating...")
        else:
            break

    print(f"\n🧭 Context-based Answer:\n{answer}")
    print(f"Hallucination: {'YES' if is_hallucinated else 'NO'} (sim={similarity:.3f})")
    print(f"Toxic: {'YES' if is_toxic else 'NO'}\n")
