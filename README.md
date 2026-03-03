
# AI-Agent-Testing
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

---
Team:
- Mohammed Mamoune Katni
- Mohamed Trombati
- Sam Hajj Assaf
