from unittest.mock import MagicMock
from bot import get_dynamic_flash_models

def test_get_dynamic_flash_models_deduplication():
    def _mk(name):
        m = MagicMock()
        m.name = f"models/{name}"
        m.supported_actions = ["generateContent"]
        return m

    client = MagicMock()
    client.models.list.return_value = [
        _mk("gemini-2.5-flash"),
        _mk("gemini-3.7-flash"),
        _mk("gemini-3-flash-preview"),
        _mk("gemini-4-carbon"),
        _mk("gemini-3.5-flash-lite"),
        _mk("gemini-3.8-flash"),
        _mk("gemini-3.8-flash"),
        _mk("gemini-4-hydrogen"),
        _mk("gemini-3.6-flash"),
    ]

    models = get_dynamic_flash_models(client)
    assert len(models) <= 5
    assert len(models) == len(set(models))
    for excluded in ("gemini-4-carbon", "gemini-3-flash-preview", "gemini-2.5-flash", "gemini-3.7-flash"):
        assert excluded not in models
    assert models.index("gemini-4-hydrogen") < models.index("gemini-3.8-flash")
    assert models.index("gemini-3.8-flash") < models.index("gemini-3.5-flash-lite")
