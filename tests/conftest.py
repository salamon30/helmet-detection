import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def client():
    # Startup event'te get_model() çağrılır — gerçek .pt dosyasını yüklemeden geçiyoruz
    with patch("api.main.get_model", return_value=MagicMock()):
        from api.main import app
        with TestClient(app) as c:
            yield c
