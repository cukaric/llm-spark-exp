from hypothesis import given
from hypothesis import strategies as st

from llm_spark_exp import DATA_DIR, MODELS_DIR, NOTEBOOKS_DIR, PROJECT_ROOT


def test_project_paths_exist() -> None:
    assert PROJECT_ROOT.exists()
    assert DATA_DIR.is_dir()
    assert MODELS_DIR.is_dir()
    assert NOTEBOOKS_DIR.is_dir()


@given(st.sampled_from([DATA_DIR, MODELS_DIR, NOTEBOOKS_DIR]))
def test_project_paths_stay_inside_repo(path) -> None:
    assert path.is_relative_to(PROJECT_ROOT)
