"""Load local literal configuration without shell evaluation or key logging."""
import os
from pathlib import Path

ALLOWED = {"AGENTLAB_API_KEY", "AGENTLAB_BASE_URL", "AGENTLAB_MODEL", "AGENTLAB_OPTIMIZER_MODEL"}


def load_local_env(path=None):
    path = Path(path) if path is not None else Path(__file__).resolve().parent.parent / ".env"
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        name, sep, value = text.partition("=")
        name, value = name.strip(), value.strip()
        if not sep or name not in ALLOWED:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if value and not os.environ.get(name):
            os.environ[name] = value
