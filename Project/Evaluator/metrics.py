from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
from nltk.tokenize import word_tokenize
import nltk

# Download necessary data (only the first time)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def compute_metrics(y_true, y_pred):
    # Tokenize inputs
    tokenized_true = [word_tokenize(t) for t in y_true]
    tokenized_pred = [word_tokenize(p) for p in y_pred]

    smoothie = SmoothingFunction().method4

    # BLEU Score
    bleu_scores = [sentence_bleu([t], p, smoothing_function=smoothie)
                for t, p in zip(tokenized_true, tokenized_pred)]
    avg_bleu = sum(bleu_scores) / len(bleu_scores) # type: ignore

    # Corrected METEOR Score line
    meteor_scores = [meteor_score([t], p)
                for t, p in zip(tokenized_true, tokenized_pred)]
    avg_meteor = sum(meteor_scores) / len(meteor_scores)

    # ROUGE Score (original sentences)
    rouge = Rouge()
    rouge_scores = rouge.get_scores(y_pred, y_true, avg=True)

    # BERTScore
    P, R, F1 = bert_score(y_pred, y_true, lang='en', rescale_with_baseline=True)

    metrics = {
        'bleu': avg_bleu,
        'meteor': avg_meteor,
        'rouge': rouge_scores,
        'bertscore': {
            'precision': P.mean().item(),
            'recall': R.mean().item(),
            'f1': F1.mean().item()
        }
    }

    return metrics

def print_metrics(metrics):
    """
    Print formatted textual evaluation metrics.

    Args:
        metrics (dict): Metrics dictionary from compute_metrics.
    """
    print("\n===== Evaluation Metrics =====")
    print(f"🔹 BLEU Score: {metrics['bleu']:.4f}")
    print(f"🔹 METEOR Score: {metrics['meteor']:.4f}\n")

    rouge = metrics['rouge']
    print("🔹 ROUGE Scores:")
    for metric, scores in rouge.items():
        print(f"  - {metric.upper()}: Precision={scores['p']:.4f}, Recall={scores['r']:.4f}, F1={scores['f']:.4f}")

    bscore = metrics['bertscore']
    print("\n🔹 BERTScore:")
    print(f"  - Precision: {bscore['precision']:.4f}")
    print(f"  - Recall:    {bscore['recall']:.4f}")
    print(f"  - F1 Score:  {bscore['f1']:.4f}")
