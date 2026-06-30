from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.api.main import app


client = TestClient(app)


def make_image() -> bytes:
    image = Image.new("RGB", (64, 64), (180, 120, 190))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_species_lookup() -> None:
    response = client.get("/species/0")
    assert response.status_code == 200
    assert response.json()["name"] == "Staphylococcus aureus"


def test_classify_image() -> None:
    response = client.post(
        "/classify",
        files={"file": ("sample.jpg", make_image(), "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["top3"]) == 3
    assert data["heatmap_url"].startswith("data:image/png;base64,")
    assert data["clinical_info"]["id"] == data["top3"][0]["class_id"]
