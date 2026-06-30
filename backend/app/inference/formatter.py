from app.data.taxonomy import TaxonomyDB


class ResultFormatter:
    def __init__(self) -> None:
        self.taxonomy = TaxonomyDB()

    def format(self, prediction: dict, heatmap_url: str) -> dict:
        class_id = prediction["predicted_class_id"]
        return {
            "top3": prediction["top3"],
            "heatmap_url": heatmap_url,
            "clinical_info": self.taxonomy.query_species(class_id),
            "focus_score": prediction["focus_score"],
            "inference_time_ms": prediction["inference_time_ms"],
            "mode": prediction["mode"],
            "disclaimer": "Academic prototype only; not for clinical diagnosis.",
        }
