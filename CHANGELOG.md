# Changelog

All notable changes to Citra are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions are
git tags.

## [Unreleased]

## [0.1.0] - 2026-09-13

The first tagged version: the assistant as it has been running in one
flat in Delhi, plus the engineering scaffolding that was missing from
the initial public push.

### Added
- A real test suite (`tests/`, 320+ tests, `pytest -q`). Every board,
  model and microphone boundary is faked in `tests/conftest.py`, so the
  suite runs on a laptop with nothing attached and in CI on Linux.
- GitHub Actions CI: ruff + pytest on Linux, `pip-audit` over every
  pinned dependency, and a full `requirements.txt` install + test run
  on Windows, the real platform.
- `citra_logging.py`: one JSON-lines logging setup shared by every
  process (timestamp, level, module, message, exceptions, extra
  fields). `CITRA_LOG_FORMAT=text` restores the human line.
- `GET /health` on the dashboard server, for a watchdog or a Pi-side
  supervisor. Touches no hardware.
- A deny-list, length cap and `CITRA_RUN_COMMAND=off` switch on the
  `run_command` tool. Everyday commands still run; formatting a drive,
  deleting a system root, wiping the registry or shutting the machine
  down from under the voice pipeline do not.
- `requirements.lock.txt`, `requirements-dev.txt`, `.env.example`,
  `CONTRIBUTING.md`, `SECURITY.md`, Dependabot and a pre-commit config.

### Changed
- `jarvis_router.py` (1694 lines) is now three modules:
  `citra_fast_path.py` (the regex-to-board table),
  `citra_llm_client.py` (LM Studio and Gemini backends with a shared
  `ConversationMemory`), and a 475-line router that owns the
  controllers and the fast-then-smart decision. No behaviour change.
- The spoken protocols and the SemanticRouter moved out of
  `jarvis_voice_assistant.py` (3714 -> 2977 lines) into
  `citra_protocols.py`, which speaks through an injected
  `set_speaker()` instead of reaching into the assistant.
- Every runtime dependency is pinned to the version Citra is run with,
  and seven packages the code imports but `requirements.txt` never
  listed (silero-vad, torch, sentence-transformers, PyAudio, av,
  pywin32, psutil) are now in it. A fresh clone installs.
- The whole tree passes `ruff check` (E, F, W, B, I, UP).

### Fixed
- `.gitignore` never actually ignored `citra_api_token.txt` or
  `citra_contacts.json` (real phone numbers): a trailing `# comment` on
  the pattern line is part of the pattern to git. Both are ignored now,
  and a test asks git itself so it cannot regress.

### Security
- See SECURITY.md for the threat model. The `run_command` deny-list and
  the `.gitignore` fix above are the two changes that matter.

[Unreleased]: https://github.com/AbhayPhalswal/citra/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/AbhayPhalswal/citra/releases/tag/v0.1.0
