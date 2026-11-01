import os
from agentlab.local_config import load_local_env


def test_local_key_loaded_and_environment_takes_precedence(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENTLAB_API_KEY", raising=False)
    p = tmp_path / ".env"
    p.write_text('AGENTLAB_API_KEY="fake-local-key"\n')
    load_local_env(p)
    assert os.environ["AGENTLAB_API_KEY"] == "fake-local-key"
    monkeypatch.setenv("AGENTLAB_API_KEY", "fake-existing-key")
    load_local_env(p)
    assert os.environ["AGENTLAB_API_KEY"] == "fake-existing-key"


def test_config_is_literal_and_only_allowed_names_loaded(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENTLAB_API_KEY", raising=False)
    monkeypatch.delenv("AGENTLAB_UNRELATED_TEST", raising=False)
    p = tmp_path / ".env"
    p.write_text('AGENTLAB_API_KEY=$(echo literal)\nAGENTLAB_UNRELATED_TEST=ignored\n')
    load_local_env(p)
    assert os.environ["AGENTLAB_API_KEY"] == "$(echo literal)"
    assert "AGENTLAB_UNRELATED_TEST" not in os.environ


def test_missing_or_blank_config_does_not_load_empty_key(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENTLAB_API_KEY", raising=False)
    p = tmp_path / ".env"
    load_local_env(p)
    p.write_text('AGENTLAB_API_KEY=\n')
    load_local_env(p)
    assert "AGENTLAB_API_KEY" not in os.environ
