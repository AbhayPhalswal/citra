# Security

Citra controls mains-wired relays and can run commands on the PC it
lives on. This is what keeps that from being a problem, and what to do
if you find something that gets past it.

## Reporting

Open an issue on the repository, or message @AbhayPhalswal on GitHub.
If it is something that should not be public yet, say so in the issue
and it will be taken to a private channel. You will get a reply.

## The threat model, briefly

**Nothing listens off this machine by default.** The dashboard server
binds to `127.0.0.1`. Binding anywhere else needs `CITRA_ALLOW_REMOTE=1`
on top of `CITRA_BIND_HOST`, and the server refuses to start without it
because the dashboard and the WebSocket carry no authentication of
their own. The intended remote path is a private mesh VPN (Tailscale),
never a port-forward.

**The phone API is token-gated.** Every `/api/*` route needs
`X-Citra-Token` (or `?token=`) matching `citra_api_token.txt`, compared
with `hmac.compare_digest`. The token is generated on first run, is
gitignored, and reaches lights and the AC only - no PC tool is on the
phone API at all.

**Board input is validated before the network.** Relay numbers,
on/off states and AC temperatures are checked in the API handlers and
in `SmartRoomController`; the firmware checks again.

**The model cannot call arbitrary code.** Tool names from the LLM are
looked up in an explicit whitelist built at import time, not
`getattr`-ed. Bad names, malformed arguments and raised exceptions
become JSON error payloads, never crashes.

**`run_command` is the deliberate exception**, and it has one guard
rail: `run_command_refusal()` in `jarvis_pc_control.py` refuses a short
deny-list of irreversible commands (formatting a drive, deleting a
system root, wiping the registry, shutting down), empty input, and
anything over 500 characters, after normalising quotes and whitespace.
`CITRA_RUN_COMMAND=off` removes the tool entirely - use that for a
household install. Every refusal and every executed command is logged.
It still runs with the privileges of the account Citra runs as; do not
run Citra as Administrator unless you need the hosts-file tools.

**Secrets never enter git.** `citra_contacts.json` (real phone numbers),
`citra_api_token.txt`, TLS keys and `wifi_secrets.h` are gitignored,
and `tests/test_gitignore_keeps_secrets_out.py` asks git itself on
every run so a broken pattern cannot silently un-ignore them again -
which is exactly what a trailing `# comment` on a pattern line once did.

**Dependencies are pinned and audited.** `requirements.txt` and
`requirements.lock.txt` carry exact versions; CI runs `pip-audit` over
both on every push and Dependabot proposes updates weekly.

## What is out of scope

Anyone who can already run code as your user, or has physical access to
the machine, can do everything Citra can. Citra does not try to defend
against its own operator.
