# Working on Citra

This is how the repository is developed. Citra is source-available, not
open source - see [LICENSE](LICENSE) - so contributions and forks need
permission first; ask by opening an issue. Everything below applies
whether you have that permission or are the maintainer.

## Set up

```bash
git clone https://github.com/AbhayPhalswal/citra
cd citra
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements-dev.txt
pytest -q
```

That is enough to run the whole test suite and the linter on any
machine, including Linux - the suite never needs a NodeMCU, a model
server, a Gemini key or a microphone. To actually *run* Citra you need
Windows and the full runtime set:

```bash
pip install -r requirements.txt   # pinned; requirements.lock.txt is the full freeze
python citra_doctor.py            # checks every subsystem in ~10s
```

Environment variables are documented in [.env.example](.env.example).

## Before you commit

```bash
ruff check .        # must be clean; `ruff check . --fix` handles most of it
pytest -q           # must be green
```

`pip install pre-commit && pre-commit install` runs the ruff check for
you on every commit. CI runs both, plus `pip-audit` and a full Windows
install, on every push.

## How tests work here

Every hardware and model boundary is replaced with a fake:

- `tests/conftest.py` has `FakeSmartRoomController`, which records every
  board call and returns a canned `HardwareResult`. Use the
  `fake_controller` / `failing_controller` fixtures.
- HTTP to a board or a model goes through a scripted fake
  `requests.Session` (see `tests/test_hardware_api.py` and
  `tests/test_citra_llm_client.py`).
- Anything that writes state next to the module (phonebook, mute,
  quiet hours, device config) is pointed at `tmp_path`.
- Timers are never waited on: `ReminderController._fire` is called
  directly.

If a test needs a real board, it belongs behind `citra_doctor.py`, not
in `tests/`. A red CI build is a real bug.

## Commits

One change per commit, with the tests that pin it in the same commit.
Formatting, refactors and features do not share a commit. The message
says *why* - the code already says what. Look at `git log` for the
house style: a short imperative first line, then the reasoning.

## Where things live

| File | What it owns |
| --- | --- |
| `jarvis_voice_assistant.py` | Wake word, speech-to-text, the spoken protocols, text-to-speech. The main process. |
| `jarvis_router.py` | Owns the controllers. Fast Path first, then the Smart Path. Tool-call dispatch. |
| `citra_fast_path.py` | The regex-to-board table that works with the internet down. |
| `citra_llm_client.py` | LM Studio and Gemini backends, tool-call round trips, conversation memory. |
| `jarvis_hardware_api.py` | `SmartRoomController`: HTTP to the relay and AC boards. |
| `citra_devices.py` | `citra_devices.json` -> boards, rooms, switches, ACs. |
| `jarvis_pc_control.py` | The Windows tools: apps, layouts, winget, hosts-file blocking, `run_command`. |
| `jarvis_vision.py` | Local screen reading and input injection. |
| `jarvis_reminders.py` | Reminders and scheduled actions. |
| `citra_contacts.py` | The phonebook and fuzzy name resolution. |
| `citra_ui_server.py` | The dashboard, WebSocket, and the token-gated phone API. |
| `citra_logging.py` | One logging setup for every process. |
| `citra_doctor.py` | Checks the whole installation against real hardware. |
| `relay_server/`, `ac_ir_server/` | ESP8266 firmware. |
