from deepeval import assert_test
from deepeval.metrics.g_eval.g_eval import GEval
from deepeval.test_case.llm_test_case import LLMTestCase
from deepeval.test_case.llm_test_case import LLMTestCaseParams




correctness_metric = GEval(
    name="Correctness",
    criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=0.5  # You can adjust this based on your tolerance
)

def run_deepeval_tests(queries, y_pred, y_true):
    for i, (query, pred, true) in enumerate(zip(queries, y_pred, y_true), 1):
        test_case = LLMTestCase(
            input=query,
            actual_output=pred,
            expected_output=true,
            # Optional: Add retrieval_context if you use RAG-based outputs
        )
        print(f"\n🔍 Running DeepEval test {i} for query: {query}")
        assert_test(test_case, [correctness_metric])

