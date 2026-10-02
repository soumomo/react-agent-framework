# ReAct Agent Engine with Context Compaction

A modular Python framework for autonomous agents that solve multi-step problems through the ReAct (Reasoning + Acting) loop. Includes a software-debugging agent (`CodeAgent`), a game-playing agent with programmatic lookahead search (`ChessAgent`), and a working-memory context compaction engine designed to cap context window growth during long trajectories.

---

## Architecture & Project Structure

```
react-agent-framework/
├── src/react_agent/
│   ├── agent/
│   │   ├── base.py          # Generic ReAct loop, token estimation, context compaction
│   │   ├── code_agent.py    # CodeAgent: bash execution, git diff generation, completion
│   │   ├── chess_agent.py   # ChessAgent: turn-based tool dispatching and game loop
│   │   ├── chess_tools.py   # Tool helpers: play_move, simulate_move, run_python
│   │   └── tools.py         # Strict OpenAI-compatible function schemas
│   ├── env.py               # Sandboxed container runtime (Modal / SWE-ReX)
│   ├── sandbox_python.py    # Remote runner for executing model-generated scripts
│   ├── chess_sandbox.py     # Sandbox deployment wrapper for the chess environment
│   ├── chess_server.py      # FastAPI chess engine exposing state, move, and simulation
│   ├── prompts.py           # Prompts and Jinja2 templates
│   └── cli.py               # CLI entry points
├── tasks/                   # Task specifications and markdown skills (SKILL.md)
├── tests/                   # Offline unit test suite
└── pyproject.toml           # Package configuration and dependencies
```

---

## Empirical Benchmarks & Results

All benchmarks below were executed using the `deepseek-chat` model backend.

### 1. SWE-bench Issue Resolution (`django__django-15368`)

The framework was evaluated on an official SWE-bench benchmark instance (`django__django-15368`). The task required inspecting a complex ORM bug where `bulk_update()` failed with an `OperationalError` when updating fields referencing `F()` expressions, writing a fix, running the Django query test suite, and exporting a clean git patch.

Both the uncompacted baseline and the working-memory compacted agent successfully resolved the bug:
- **Model**: `deepseek-chat`
- **FAIL_TO_PASS**: `test_f_expression` passed (1/1).
- **PASS_TO_PASS**: All 29 regression tests passed (29/29).
- **Final Result**: **RESOLVED** :D

---

### 2. Context Compaction Efficiency

To measure context efficiency, the identical SWE-bench task was run under two conditions with `deepseek-chat`:
1. **Baseline**: Conversation history preserved verbatim across all steps.
2. **Context Compaction**: Intermediate history older than the most recent step is summarized into a structured `<working_memory>` block whenever token count exceeds the threshold.

| Metric | Baseline (Uncompacted) | With Context Compaction | Improvement |
| :--- | :--- | :--- | :--- |
| **Model** | `deepseek-chat` | `deepseek-chat` | — |
| **Task Status** | RESOLVED | **RESOLVED** | Parity |
| **Total ReAct Steps** | 29 steps | **28 steps** | -1 step |
| **Compaction Events** | 0 | **10** | — |
| **Peak Prompt Tokens** | 35,720 tokens | **4,944 tokens** | **-86.2% peak size** |
| **Final Prompt Tokens** | 35,720 tokens | **4,872 tokens** | **-86.4% smaller context** |
| **Total Cumulative Tokens** | 641,076 tokens | **87,985 tokens** | **-86.3% token savings** :) |

**Observation**: Uncompacted history grows monotonically, making later steps slow and expensive. Compaction produces a stable "sawtooth" token trajectory that keeps prompt size bounded below 5,000 tokens while preserving critical state (active files, test outcomes, next steps). That saved over 550,000 tokens on a single debugging task :)

---

### 3. Programmatic Search vs. Sequential Tool Calls

In the chess environment, rather than forcing the model to make dozens of sequential single-move API round-trips over the internet, `ChessAgent` can write Python code containing lookahead search logic:
- The script is base64-encoded and sent to `/opt/assignment/sandbox_python.py`.
- Inside the sandbox, `simulate_move(fen, move)` evaluates prospective positions against the server over `localhost`.
- Once the best move is found via local minimax/heuristic evaluation, the script calls `play_move(best)` to commit it to the live match.

---

## Setup & Execution

### 1. Installation
Install dependencies using [`uv`](https://docs.astral.sh/uv/):
```bash
uv sync
```

Run the unit tests:
```bash
uv run pytest
```

### 2. Configuration
Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
OPENAI_MAX_RETRIES=5
```

Verify your configuration and endpoint connectivity:
```bash
uv run agent-doctor
```

---

## CLI Usage

### Run the Code Debugging Agent
```bash
uv run agent-code \
  --task tasks/chess-terminal-move \
  --skills-path tasks/code-skills \
  --trajectory artifacts/part1-trajectory.json \
  --patch-output artifacts/fix.patch
```

### Run SWE-bench with Context Compaction
```bash
uv run agent-swebench django__django-15368 \
  --compact-threshold-tokens 6000 \
  --trajectory artifacts/django-compacted-trajectory.json \
  --patch-output artifacts/django-fix.patch
```

### Run the Chess Agent
**Direct play:**
```bash
uv run agent-chess \
  --task tasks/chess-terminal-move \
  --patch artifacts/fix.patch \
  --trajectory artifacts/chess-trajectory.json \
  --result artifacts/chess-result.json
```

**With programmatic search and strategy skills:**
```bash
uv run agent-chess \
  --task tasks/chess-terminal-move \
  --patch artifacts/fix.patch \
  --programmatic-tools \
  --skills-path tasks/chess-skills \
  --trajectory artifacts/chess-python-trajectory.json
```

---

## Troubleshooting & Runtime Notes

- **Remote Tunnel Timeouts (`httpx.RemoteProtocolError`)**: When running long games over remote container tunnels (Modal), transient socket drops can occur. The HTTP helper (`_request_state`) includes automatic 3x retries to handle connection resets gracefully.
- **Provider Rate Limits (`429`)**: If using free-tier model providers capped at low requests-per-minute (RPM), configure exponential backoff in `base.py` and pass `--step-limit` appropriately.
- **Sandboxed Execution**: Model-written scripts execute strictly inside the container runner (`sandbox_python.py`) over localhost and never execute in the local host agent process.

---

## License
MIT
