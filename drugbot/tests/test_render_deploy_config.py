from pathlib import Path

from app.core.config import settings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent


def test_render_uses_project_subdir_and_pythonpath():
    render_text = (REPO_ROOT / "render.yaml").read_text(encoding="utf-8")
    assert "rootDir: drugbot" in render_text
    assert "PYTHONPATH" in render_text
    assert "python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT" in render_text
    assert "ENABLE_BM25_STARTUP_BUILD" in render_text


def test_requirements_include_runtime_dependencies():
    requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "pymupdf" in requirements.lower()
    assert "sqlalchemy" in requirements.lower()
    assert "bcrypt" in requirements.lower()
    assert "pyjwt" in requirements.lower()


def test_database_paths_are_project_absolute():
    sqlite_path = Path(settings.sqlite_db_path)
    chroma_path = Path(settings.chroma_db_path)
    assert sqlite_path.is_absolute()
    assert chroma_path.is_absolute()
    assert sqlite_path.name == "app.db"
    assert chroma_path.name == "chroma_db"
