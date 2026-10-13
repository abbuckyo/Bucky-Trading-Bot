import os
import pytest
from pathlib import Path
from bot import get_master_password, encrypt_payload, write_encrypted_dashboard, selftest_decrypt


def test_get_master_password_missing(monkeypatch):
    monkeypatch.delenv("DASHBOARD_PASSWORD", raising=False)
    with pytest.raises(ValueError, match="DASHBOARD_PASSWORD environment variable is not set"):
        get_master_password()


def test_get_master_password_empty(monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSWORD", "   ")
    with pytest.raises(ValueError, match="DASHBOARD_PASSWORD environment variable is not set"):
        get_master_password()


def test_get_master_password_valid(monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSWORD", "  MySecretPass123!  ")
    assert get_master_password() == "MySecretPass123!"


def test_encrypt_payload_without_env(monkeypatch):
    monkeypatch.delenv("DASHBOARD_PASSWORD", raising=False)
    payload = {"assets": []}
    with pytest.raises(ValueError, match="DASHBOARD_PASSWORD environment variable is not set"):
        encrypt_payload(payload)


def test_encrypt_payload_with_env(monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSWORD", "Secret123")
    payload = {"assets": ["A", "B"]}
    envelope = encrypt_payload(payload)
    assert envelope["v"] == 2
    assert envelope["alg"] == "AES-GCM"
    # Should be able to decrypt with the same password
    selftest_decrypt(envelope, "Secret123", payload)


def test_write_encrypted_dashboard_without_env(monkeypatch):
    monkeypatch.delenv("DASHBOARD_PASSWORD", raising=False)
    payload = {"assets": []}
    with pytest.raises(ValueError, match="DASHBOARD_PASSWORD environment variable is not set"):
        write_encrypted_dashboard(payload)


def test_write_encrypted_dashboard_with_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DASHBOARD_PASSWORD", "Secret123")
    monkeypatch.chdir(tmp_path)
    payload = {"assets": ["A"]}
    out_path = write_encrypted_dashboard(payload)
    assert out_path == Path("docs/data.enc")
    assert out_path.exists()
