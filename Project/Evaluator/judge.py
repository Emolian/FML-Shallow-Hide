from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def compute_rouge_scores(reference, generated):
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L scores.

    Parameters:
        reference (str): The reference/gold answer.
        generated (str): The model-generated answer.

    Returns:
        dict: A dictionary of ROUGE scores.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return {k: v.fmeasure for k, v in scores.items()}


def compute_bleu_score(reference, generated):
    """
    Compute BLEU score for a single sentence pair.

    Parameters:
        reference (str): The reference answer.
        generated (str): The model-generated answer.

    Returns:
        float: BLEU score between 0 and 1.
    """
    ref_tokens = reference.lower().split()
    gen_tokens = generated.lower().split()
    smoothing = SmoothingFunction().method4
    return sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoothing)


def evaluate_generated_answer(reference, generated):
    """
    Compute and print evaluation metrics (ROUGE and BLEU).

    Parameters:
        reference (str): Ground-truth reference answer.
        generated (str): Generated answer from the model.
    """
    rouge_scores = compute_rouge_scores(reference, generated)
    bleu_score = compute_bleu_score(reference, generated)

    print("--- Evaluation ---")
    print(f"BLEU Score: {bleu_score:.4f}")
    for k, v in rouge_scores.items():
        print(f"{k.upper()}: {v:.4f}")