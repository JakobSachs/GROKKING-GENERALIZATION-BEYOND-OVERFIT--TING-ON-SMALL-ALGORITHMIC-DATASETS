"""The fixed upstream experiment and its provenance, shared by both CLIs."""

from hashlib import sha256
from pathlib import Path

MODEL_CONFIG = {
    "syms": 63,
    "maxlen": 3,
    "layers": 2,
    "embed_dim": 128,
    "num_heads": 4,
    "ff_dim": 512,
}
MODULUS = 62
UPSTREAM_COMMIT = "5fe20591e86bdde7a2f4323ee0225e28b69ceb39"
SOURCE_HASHES = {
    "transf.py": "97a7e02934f0709d0ce389a44dac3654a5ef9251ccd6035f00d42a4aced83bca",
    "grokking.py": "32a946ad312370621e974e0a90c92f5248dfe298d23910dd138f66b5129495d9",
}
ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def verify_sources() -> None:
    for filename, expected in SOURCE_HASHES.items():
        if digest(ROOT / filename) != expected:
            raise ValueError(f"upstream source changed: {filename}")
