from datasets import load_dataset
def accuracy_reward_function(predictions, ground_truth):
    """
    Accuracy-based reward function: R = (1/N) * Σ(I(ŷᵢ = yᵢ))
    Args:
        predictions: List of model predictions (strings)
        ground_truth: List of ground truth answers (strings)
    Returns:
        Float reward score between 0 and 1 (accuracy)
    """
    if not predictions or not ground_truth:
        return 0.0
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have the same length")
    N = len(predictions)
    correct_predictions = sum(pred == true_label for pred, true_label in zip(predictions, ground_truth))
    return correct_predictions / N

def get_elmb_questions_and_answers(split="train", num_samples=None):
    """
    Loads the ELMB-Reasoning dataset and returns questions and correct answers.
    Args:
        split: Dataset split to use ("train", "test", etc.)
        num_samples: If set, only return this many samples
    Returns:
        questions: List of question strings
        answers: List of correct answer strings
    """
    ds = load_dataset("data4elm/ELMB-Reasoning", split=split)
    if num_samples:
        ds = ds.select(range(num_samples))
    questions = ds["question"]
    answers = ds["answer"]
    return questions, answers

if __name__ == "__main__":
    # Load questions and correct answers from ELMB-Reasoning
    questions, ground_truth = get_elmb_questions_and_answers(num_samples=10)

    # Simulate model predictions (for demonstration, use ground truth or random)
    # In practice, replace this with your model's outputs
    predictions = []
    for q in questions:
        # Example: echo the correct answer (perfect accuracy)
        # Replace this with: model.generate(q)
        predictions.append("dummy_prediction")  # Replace with actual model output

    # For demonstration, let's use the correct answers as predictions (perfect accuracy)
    # predictions = ground_truth

    # Calculate accuracy
    accuracy = accuracy_reward_function(predictions, ground_truth)
    print(f"ELMB Function Calling Accuracy Reward: {accuracy:.3f}")

    # Show a few examples
    for i in range(min(3, len(questions))):
        print(f"\nQ: {questions[i]}")
        print(f"Model Prediction: {predictions[i]}")
        print(f"Correct Answer: {ground_truth[i]}")
