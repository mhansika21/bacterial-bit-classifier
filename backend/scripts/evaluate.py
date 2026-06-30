import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained bacterial classifier checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    print("Evaluation entry point")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Manifest: {args.manifest}")
    print("Add the held-out test manifest to compute accuracy, macro F1, AUC-ROC, and confusion matrix.")


if __name__ == "__main__":
    main()
