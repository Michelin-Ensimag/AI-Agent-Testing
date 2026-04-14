
# AI-Agent-Testing
[![CI](https://github.com/Michelin-Ensimag/AI-Agent-Testing/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Michelin-Ensimag/AI-Agent-Testing/actions/workflows/ci.yml) [![DeepEval Report](https://img.shields.io/website?url=https%3A%2F%2Fmichelin-ensimag.github.io%2FAI-Agent-Testing%2F&up_message=live&down_message=offline&label=Allure%20Report)](https://michelin-ensimag.github.io/AI-Agent-Testing/)

Repository for the project "Tester les Agents IA"


## Setup

**First time** — install [uv](https://docs.astral.sh/uv/) if you don't have it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh  # restart terminal after
```

**After every `git pull`:**
```bash
uv sync
```
```bash
source .venv/bin/activate  # VSCode usually does this automatically
```

**Adding a package:**
```bash
uv add <package>       # normal dependency
uv add --dev <package> # dev/test only (e.g. linters, pytest plugins)
```
Then commit `pyproject.toml` and `uv.lock`.
