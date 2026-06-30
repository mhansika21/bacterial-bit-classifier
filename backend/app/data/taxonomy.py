from app.data.species import SPECIES


class TaxonomyDB:
    def __init__(self) -> None:
        self._by_id = {item["id"]: item for item in SPECIES}

    def get_all(self) -> list[dict]:
        return list(self._by_id.values())

    def query_species(self, species_id: int) -> dict | None:
        return self._by_id.get(species_id)
