# AGENTS.md

## Project

- Educational implementations of 4 agentic patterns (reflection, tool use, ReAct planning, multiagent) using direct Groq API calls. No LangChain/LangGraph/CrewAI.
- `src/agentic_patterns/` is the installable `agentic-patterns` package; `notebooks/` contains the step-by-step lessons and `README.md` the course material. Notebooks import the package but also define example tools/loops inline; do not refactor lesson code into imports.
- No test suite, lint config, or type checker. CI only builds and publishes to PyPI on pushes to `main`.

## Setup

- `poetry install` (Python `^3.11`), then copy `.env.example` to `.env` and set `GROQ_API_KEY`.
- Notebooks import `agentic_patterns`, so the package must be installed (or use `PYTHONPATH=src`).
- Root `uv.lock` is a stub (`requires-python >=3.14`, no packages); `uv sync` creates an empty environment. Use Poetry.
- If `.venv/` is the Windows venv in this checkout, use `.venv/Scripts/python.exe`; there is no `.venv/bin/python`.

## Commands

- Format/lint: `poetry run pre-commit run --all-files` (black, reorder-python-imports with `--application-directories=src`, trailing whitespace).
- There are no tests to run. Verify changes with focused scripts or notebook cells. Every agent call hits Groq, so keep `n_steps`/`max_rounds` small.

## Architecture

- `utils/completions.py` is the only Groq wrapper plus prompt/history helpers. `ChatHistory` drops messages from the front; `FixedFirstChatHistory` keeps the system prompt fixed.
- Pattern packages: `reflection_pattern/`, `tool_pattern/`, `planning_pattern/`, `multiagent_pattern/`.
- `agentic_patterns` exports only `ReflectionAgent`; import every other class from its submodule.
- `multiagent_pattern.agent.Agent` wraps `ReactAgent`. Construct agents inside `with Crew() as crew:` or they never register. `agent_a >> agent_b` declares dependencies; `crew.run()` topologically sorts, prints outputs, and returns `None`.
- Default model `llama-3.3-70b-versatile` is hardcoded in `reflection_agent.py`, `tool_agent.py`, `react_agent.py`, and `multiagent_pattern/agent.py`; update all four when changing models. `notebooks/reflection_pattern.ipynb` still hardcodes the deprecated `llama3-70b-8192`.

## Tool and loop quirks

- `@tool` is not a normal decorator: it executes at import time and returns a `Tool` object, so the decorated name is not callable.
- Tool schemas use each parameter annotation's `__name__`; use only `int`, `str`, `bool`, `float`. `Optional[...]` or `X | None` produces `"Union"` and breaks `validate_arguments`.
- ReAct depends on exact tags in model output: `<thought>`, `<tool_call>{json}</tool_call>`, `<response>`. Keep prompts and `extract_tag_content` calls in sync.
- `ReflectionAgent` stops early when a critique contains `<OK>`; `n_steps` counts generate/reflect cycles.

## Git

- CRLF/LF mismatch: `git status` shows many modified files with no real changes. Use `git diff --ignore-space-at-eol` to find actual edits and stage explicit paths; never `git add -A`.
