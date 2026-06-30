import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetItem:
    file_path: Path
    label: str


class BacterialDatasetManifest:
    def __init__(self, manifest_path: str | Path, data_root: str | Path | None = None) -> None:
        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root) if data_root else self.manifest_path.parent
        self.items = self._load()

    def _load(self) -> list[DatasetItem]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        items: list[DatasetItem] = []
        with self.manifest_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"file_path", "label"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("Manifest must contain file_path and label columns")
            for row in reader:
                path = Path(row["file_path"])
                if not path.is_absolute():
                    path = self.data_root / path
                items.append(DatasetItem(path, row["label"]))
        return items

    def class_map(self) -> dict[str, int]:
        labels = sorted({item.label for item in self.items})
        return {label: idx for idx, label in enumerate(labels)}
