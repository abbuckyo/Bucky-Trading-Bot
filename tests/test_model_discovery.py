from unittest.mock import MagicMock
from bot import get_dynamic_flash_models

def test_get_dynamic_flash_models_deduplication():
    # Mock model objects returned by client.models.list()
    m1 = MagicMock()
    m1.name = "models/gemini-2.5-flash"
    m1.supported_actions = ["generateContent"]

    m2 = MagicMock()
    m2.name = "models/gemini-3.5-flash"
    m2.supported_actions = ["generateContent"]

    m3 = MagicMock()
    m3.name = "models/gemini-3.5-flash"
    m3.supported_actions = ["generateContent"]

    client = MagicMock()
    client.models.list.return_value = [m1, m2, m3]

    models = get_dynamic_flash_models(client)
    # gemini-3.5-flash (version 3.5) should come before gemini-2.5-flash (version 2.5), and duplicates are removed preserving order
    assert models == ["gemini-3.5-flash", "gemini-2.5-flash"]
