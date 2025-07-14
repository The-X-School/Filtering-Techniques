import torch
from Model_filter_data import ModelBasedFilter
from datasets import Dataset

def test_maki_model():
    print("🚀 Testing Maki/Llama-3.2-1B-reasoning model")
    print("=" * 50)

    model_name = "MakiLS/Llama-3.2-1B-reasoning"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        print(f"Initializing ModelBasedFilter with model: {model_name} on device: {device}")
        filter_system = ModelBasedFilter(
            model_name=model_name,
            device=device
        )
        filter_system.quality_threshold = 0.5  # Set a default threshold for testing

        sample_texts = [
            "The sum of two even numbers is always an even number. For example, 2 + 4 = 6, and 6 is an even number. This holds true for any pair of even integers.",
            "If all birds can fly, and a penguin is a bird, then a penguin can fly. This statement is logically flawed because the premise 'all birds can fly' is false.",
            "The capital of France is Berlin. This is a factual error.",
            "1 + 1 = 3. This is incorrect mathematical reasoning.",
            "A square has four equal sides and four right angles. Therefore, a rectangle is always a square. This is incorrect because a rectangle only requires four right angles, not necessarily four equal sides.",
            "The quick brown fox jumps over the lazy dog.", # Neutral text
            "This text is very bad and contains many errors and is not good for anything."
        ]

        print("\nCalculating quality scores for sample texts:")
        results = []
        for i, text in enumerate(sample_texts):
            print(f"\n--- Sample {i+1} ---")
            print(f"Text: {text[:100]}...")
            quality_result = filter_system.calculate_quality_score(text)
            results.append(quality_result)
            print(f"  Quality Score: {quality_result['score']:.3f}")
            print(f"  Perplexity: {quality_result['perplexity']:.2f}")
            print(f"  Reason: {quality_result['reason']}")
            if 'error' in quality_result:
                print(f"  Error: {quality_result['error']}")

        print("\n\nFiltering a small dataset:")
        sample_dataset = Dataset.from_list([{"text": t} for t in sample_texts])
        filtered_samples = filter_system.filter_dataset(sample_dataset)

        print(f"\nOriginal samples: {len(sample_texts)}")
        print(f"Filtered samples: {len(filtered_samples)}")
        print(f"Retention rate: {len(filtered_samples)/len(sample_texts)*100:.1f}%")

        print("\nSample filtered results:")
        for i, sample in enumerate(filtered_samples):
            print(f"\nSample {i+1}:")
            print(f"  Text: {sample['text'][:80]}...")
            print(f"  Quality Score: {sample['quality_analysis']['score']:.3f}")
            print(f"  Perplexity: {sample['quality_analysis']['perplexity']:.2f}")

        print("\n" + "=" * 50)
        print("✅ Maki/Llama-3.2-1B-reasoning model test completed!")

    except Exception as e:
        print(f"❌ Error during model testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_maki_model()
