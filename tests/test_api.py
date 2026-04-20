from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_debug_routes():
    print([route.path for route in app.routes])
    assert "/health" in [route.path for route in app.routes]
