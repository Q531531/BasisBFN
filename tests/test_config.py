import pytest

from sgbformer.config import load_config


def test_config_merges_nested_fields_and_expands_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("BASISBFN_TEST_ROOT", "/test/data")
    (tmp_path / "base.yaml").write_text("model:\n  a: 1\n  b: 2\ndata: ${BASISBFN_TEST_ROOT}\n")
    path = tmp_path / "arm.yaml"
    path.write_text("_base_: base.yaml\nmodel:\n  b: 3\n")
    assert load_config(path) == {"model": {"a": 1, "b": 3}, "data": "/test/data"}


def test_config_inheritance_cycle_reports_chain(tmp_path):
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_text("_base_: b.yaml\n")
    b.write_text("_base_: a.yaml\n")
    with pytest.raises(ValueError, match="inheritance cycle") as error:
        load_config(a)
    assert "a.yaml" in str(error.value) and "b.yaml" in str(error.value)


def test_config_rejects_non_mapping_root(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text("- item\n")
    with pytest.raises(ValueError, match="root must be a mapping"):
        load_config(path)
