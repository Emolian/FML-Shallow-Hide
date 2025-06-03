# 🧠 FML-Shallow-Hide

**FML-Shallow-Hide** is an AI-powered application designed to analyze and process philosophical arguments using semantic embeddings and vector search techniques. It uses a LLaMA 3.2B model and SentenceTransformers for embedding generation, and FAISS for efficient similarity search.

## 🚀 Features

- 📄 Reads and cleans philosophical argument datasets (CSV)
- ✂️ Chunks texts into manageable units
- 🔍 Generates vector embeddings using SentenceTransformers
- 🔎 Performs semantic search using FAISS
- 📊 Includes visualization tools and evaluation metrics
- 🧱 Modular pipeline for flexibility and scalability

## 🛠 Technologies Used

- Python 3
- LLaMA 3.2B
- SentenceTransformers
- FAISS (Facebook AI Similarity Search)
- Pandas, NumPy
- TensorFlow, Scikit-learn
- Matplotlib, Seaborn
- YAML for configuration

## 🗂 Project Structure

```
Project/
├── Config/           # Configuration files (YAML)
├── Embeddings/       # Precomputed embedding storage
├── Evaluator/        # Evaluation metrics and utilities
├── Guardrails/       # Rule-based filtering or validation
├── Pipeline/         # Main philosophy_pipeline.py logic
├── Runner/           # Model wrapper and execution script
├── Tests/            # Unit tests
└── Data/             # Input CSV datasets (not included in repo)
```

## 📌 Notes

- You must download or configure access to the LLaMA 3.2B model manually.
- FAISS index and metadata are saved for reuse.
- Preprocessing and chunking are controlled via the config file.

## 👨‍💻 Author

Pal Ioan Emilian
Mihalcea Andrei-Cristian 
Bachelor Students in Artificial Intelligence – Babeș-Bolyai University

## 📄 License

This project is licensed for educational and research use.
