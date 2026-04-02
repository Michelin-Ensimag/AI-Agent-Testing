"""Safeguard tests to prevent paid model usage in agent files."""

import ast
from pathlib import Path

ALLOWED_FREE_MODELS = {
    "gpt-5-mini",
    "gpt-4.1",
    "gpt-4o",
    "oswe-vscode-prime",  # VSCode Raptor mini (Preview)
}

AGENTS_ROOT = Path(__file__).resolve().parents[2] / "agents"
PYTHON_FILE_GLOB = "*.py"

# Keep this empty to scan only top-level files in src/agents.
# Add folder names here (for example "agents_demo") when you want to include them.
INCLUDED_AGENT_SUBFOLDERS: list[str] = []


def _scan_directories() -> list[Path]:
    directories = [AGENTS_ROOT]
    directories.extend(AGENTS_ROOT / folder for folder in INCLUDED_AGENT_SUBFOLDERS)
    return directories


def _discover_agent_files() -> list[Path]:
    files: list[Path] = []
    for directory in _scan_directories():
        if not directory.exists():
            raise AssertionError(
                f"Configured scan directory does not exist: {directory}"
            )

        files.extend(
            sorted(path for path in directory.glob(PYTHON_FILE_GLOB) if path.is_file())
        )
    return files


def _module_level_string_constants(tree: ast.AST) -> dict[str, str]:
    constants: dict[str, str] = {}
    for node in getattr(tree, "body", []):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            constants[node.targets[0].id] = node.value.value
    return constants


def _resolve_model_value(node: ast.AST, constants: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    return None


def _extract_declared_models(file_path: Path) -> tuple[set[str], bool]:
    tree = ast.parse(file_path.read_text(encoding="utf-8"))
    constants = _module_level_string_constants(tree)

    models: set[str] = set()
    unresolved_model_value = False

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg not in {"model", "model_name"}:
                continue

            model_value = _resolve_model_value(keyword.value, constants)
            if model_value is None:
                unresolved_model_value = True
            else:
                models.add(model_value)

    return models, unresolved_model_value


def test_agents_use_only_allowed_free_models():
    agent_files = _discover_agent_files()

    assert agent_files, (
        "No agent files found to scan. "
        f"Checked pattern '{PYTHON_FILE_GLOB}' in: "
        f"{', '.join(str(path) for path in _scan_directories())}"
    )

    failures: list[str] = []
    skipped_files: list[str] = []

    for file_path in agent_files:
        relative_path = file_path.relative_to(AGENTS_ROOT)
        models, unresolved = _extract_declared_models(file_path)

        if unresolved:
            failures.append(
                f"{relative_path}: could not statically resolve one or more model values "
                "(use a string literal or module-level string constant)."
            )
            continue

        if not models:
            skipped_files.append(
                f"{relative_path}: no model/model_name argument found, skipping file."
            )
            continue

        disallowed_models = sorted(
            model_name for model_name in models if model_name not in ALLOWED_FREE_MODELS
        )
        if disallowed_models:
            failures.append(
                f"{relative_path}: disallowed model(s) {disallowed_models}. "
                f"Allowed models: {sorted(ALLOWED_FREE_MODELS)}"
            )

    if skipped_files:
        print("Model safeguard skipped files with no model usage:")
        for message in skipped_files:
            print(f"- {message}")

    assert not failures, "Model safeguard failed:\n" + "\n".join(
        f"- {message}" for message in failures
    )
