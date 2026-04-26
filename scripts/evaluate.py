"""
Evaluate the fine-tuned model on the test set.
Reports accuracy and shows classification results.
"""
import sys
import os
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.inference import IntentClassification


def evaluate(config_path="configs/inference.yaml", test_path="sample_data/test.csv", max_samples=None):
    """
    Evaluate model on test set.
    
    Args:
        config_path: Path to inference config YAML
        test_path: Path to test CSV file
        max_samples: Limit number of test samples (None = all)
    """
    # Load test data
    df_test = pd.read_csv(test_path)
    if max_samples:
        df_test = df_test.head(max_samples)
    
    print("=" * 60)
    print("📊 BANKING INTENT - TEST SET EVALUATION")
    print("=" * 60)
    print(f"   Test samples: {len(df_test)}")
    print(f"   Config: {config_path}")
    print()
    
    # Load model
    classifier = IntentClassification(model_path=config_path)
    
    # Predict
    y_true = []
    y_pred = []
    
    print("\n📊 Running evaluation...")
    for idx, row in df_test.iterrows():
        pred = classifier(row['text'])
        y_true.append(row['intent'])
        y_pred.append(pred)
        
        if (idx + 1) % 50 == 0:
            current_acc = accuracy_score(y_true, y_pred)
            print(f"   [{idx+1}/{len(df_test)}] Current accuracy: {current_acc:.4f}")
    
    # Results
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n{'=' * 60}")
    print(f"🎯 FINAL ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'=' * 60}")
    
    # Classification report (top 20 classes)
    all_labels = sorted(set(y_true + y_pred))
    print(f"\n📋 Classification Report (showing {min(20, len(all_labels))} classes):")
    print(classification_report(y_true, y_pred, labels=all_labels[:20], zero_division=0))
    
    # Save results
    results_df = pd.DataFrame({
        "text": df_test["text"].values[:len(y_true)],
        "true_intent": y_true,
        "predicted_intent": y_pred
    })
    results_df["correct"] = results_df["true_intent"] == results_df["predicted_intent"]
    results_df.to_csv("sample_data/test_results.csv", index=False)
    print(f"✅ Detailed results saved to: sample_data/test_results.csv")
    
    return accuracy


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate banking intent model")
    parser.add_argument("--config", default="configs/inference.yaml", help="Path to inference config")
    parser.add_argument("--test", default="sample_data/test.csv", help="Path to test CSV")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit test samples")
    args = parser.parse_args()
    
    evaluate(config_path=args.config, test_path=args.test, max_samples=args.max_samples)
