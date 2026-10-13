import json
import pytest
from bot import (
    PBKDF2_ITERATIONS,
    derive_key,
    encrypt_payload,
    selftest_decrypt,
    MASTER_PASSWORD,
)


def test_pbkdf2_iterations_recommendation():
    """Verify PBKDF2 iteration count meets OWASP recommendation (>= 600,000)."""
    assert PBKDF2_ITERATIONS >= 600_000, (
        f"PBKDF2 iterations ({PBKDF2_ITERATIONS}) should be at least 600,000 "
        "to prevent brute-force attacks."
    )


def test_derive_key():
    """Verify key derivation function produces correct length bytes."""
    salt = b"0123456789abcdef"
    key = derive_key("test_password", salt, iterations=PBKDF2_ITERATIONS)
    assert isinstance(key, bytes)
    assert len(key) == 32


def test_encrypt_payload_and_selftest():
    """Verify encrypt_payload outputs correct metadata and decrypts successfully."""
    payload = {
        "version": 3.1,
        "assets": [
            {"fund": "SCBWORLDE", "tranche_stage": 1},
            {"fund": "SCBS&P500E", "tranche_stage": 0},
        ],
    }
    envelope = encrypt_payload(payload, password=MASTER_PASSWORD)

    assert envelope["v"] == 2
    assert envelope["alg"] == "AES-GCM"
    assert envelope["kdf"] == "PBKDF2-SHA256"
    assert envelope["iterations"] == 600_000
    assert "salt" in envelope
    assert "iv" in envelope
    assert "ciphertext" in envelope

    # Verify self-test decryption passes without exception
    selftest_decrypt(envelope, MASTER_PASSWORD, payload)
