import argparse
from pathlib import Path

from app.data.dataset import BacterialDatasetManifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BiT bacterial classifier")
    parser.add_argument("--manifest", required=True, help="CSV with file_path,label columns")
    parser.add_argument("--data-root", default=None, help="Root directory for relative manifest paths")
    parser.add_argument("--config", default="../configs/bit_m_r50x1.yaml")
    args = parser.parse_args()

    manifest = BacterialDatasetManifest(args.manifest, args.data_root)
    class_map = manifest.class_map()
    print(f"Loaded {len(manifest.items)} images across {len(class_map)} classes")

    missing = [str(item.file_path) for item in manifest.items if not item.file_path.exists()]
    if missing:
        raise SystemExit(f"Manifest references {len(missing)} missing files; first: {missing[0]}")

    try:
        import torch  # noqa: F401
        import timm  # noqa: F401
    except ImportError:
        print("ML dependencies are not installed. Install requirements-ml.txt to run full training.")
        return

    print("Training hook is ready. Add your dataset-specific DataLoader to launch full BiT fine-tuning.")
    print(f"Config: {Path(args.config).resolve()}")


if __name__ == "__main__":
    main()
