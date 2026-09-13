"""
DEVICE REGISTRY - which board, which channel, in which room.

WHY THIS EXISTS:

Until now the whole system assumed one relay board and one AC unit. That
assumption is spread across six files: jarvis_hardware_api validates
relay_number in (1,2,3,4), citra_ui_server has ALL_RELAY_NUMBERS, the
voice protocols default to [1,2,3,4] for "the lights", the dashboard
renders exactly four sliders, and the Gemini tool schemas declare
enum: [1,2,3,4]. Going to eleven boards by widening all of those to 1-44
would be the obvious move and the wrong one, for two reasons.

First, nobody says "turn on light 23". They say "the bedroom fan". A flat
numbering scheme is unusable by voice, which is the entire product.

Second, a 44-value enum in the tool schema is 44 values Gemini has to
reason about on every call, in a schema that is already 4,500 tokens.

So switches are addressed by ROOM and NAME, resolved here to a (host,
channel) pair. Adding a board becomes editing citra_devices.json - no
code change, no schema change, nothing to re-flash but the new board.

BOARDS ARE THE UNIT, ROOMS ARE A LABEL. An earlier version of this file
keyed everything by room, one board each, which fell apart the moment
the real layout arrived: the hall has THREE switchboards (one for the
TV, two for fans and lights) and every bedroom has two (lights on one,
fan on the other). A room is therefore a grouping over boards, not a
board - "turn off the hall" fans out across all three of its boards,
while each board keeps its own address.

FILE FORMAT (citra_devices.json, next to this file):

    {
      "boards": {
        "hall-tv":     {"host": "192.168.0.20", "room": "hall",
                        "switches": {"1": "tv"}},
        "hall-lights": {"host": "192.168.0.21", "room": "hall",
                        "switches": {"1": "main light", "2": "corner light"}}
      },
      "acs": {
        "bedroom 1": {"host": "192.168.0.11"}
      }
    }

The board key ("hall-tv") is a name for you and the logs; users never say
it. Channel keys are strings because JSON object keys always are, and are
coerced to int on load. A channel that exists physically but drives
nothing is left out rather than named "spare" - absent means "no switch
here", which is what the UI and voice layers want to know.
"""

import difflib
import json
import os
from dataclasses import dataclass

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "citra_devices.json")

# The ESP8266 firmware exposes /relay1../relay8; anything outside that is
# a config typo rather than something to attempt and let fail at the board.
MIN_CHANNEL = 1
MAX_CHANNEL = 8

# How close a spoken name has to be to count as a match. Tuned by the same
# reasoning as the voice assistant's own thresholds: high enough that
# "bedroom" does not match "bathroom" (those score ~0.67 against each
# other), low enough to absorb Whisper dropping or mangling a syllable.
NAME_MATCH_THRESHOLD = 0.72


class DeviceConfigError(Exception):
    """Raised for a malformed or contradictory citra_devices.json."""


@dataclass(frozen=True)
class Switch:
    board: str      # the board key, e.g. "hall-lights" - for logs, not for users
    room: str       # what a person calls the place, e.g. "hall"
    channel: int
    name: str
    host: str

    @property
    def label(self) -> str:
        """How a human refers to this switch: 'hall fan', 'bedroom 1 light'."""
        return f"{self.room} {self.name}"


@dataclass(frozen=True)
class AirConditioner:
    room: str
    host: str


class DeviceRegistry:
    """
    Everything the rest of the system needs to know about what is on the
    wall. Constructed once at startup and treated as immutable.
    """

    def __init__(self, boards: dict[str, dict], acs: dict[str, dict]) -> None:
        self._switches: list[Switch] = []
        self._acs: list[AirConditioner] = []
        self._board_hosts: dict[str, str] = {}
        self._board_rooms: dict[str, str] = {}

        seen_hosts: dict[str, str] = {}
        for board, spec in boards.items():
            host = spec.get("host")
            if not host:
                raise DeviceConfigError(f"board {board!r} has no host")
            room = spec.get("room")
            if not room:
                raise DeviceConfigError(f"board {board!r} has no room")

            # A duplicated host is the single most likely config mistake -
            # copy a board block, forget to change the IP - and it fails in
            # the most confusing way possible: the wrong board's switches
            # respond. Cheaper to refuse to start.
            if host in seen_hosts:
                raise DeviceConfigError(
                    f"boards {seen_hosts[host]!r} and {board!r} both use host {host} - "
                    "each board needs its own address"
                )
            seen_hosts[host] = board
            self._board_hosts[board] = host
            self._board_rooms[board] = room

            for raw_channel, name in (spec.get("switches") or {}).items():
                try:
                    channel = int(raw_channel)
                except (TypeError, ValueError) as exc:
                    raise DeviceConfigError(
                        f"board {board!r} has a non-numeric channel key {raw_channel!r}"
                    ) from exc
                if not (MIN_CHANNEL <= channel <= MAX_CHANNEL):
                    raise DeviceConfigError(
                        f"board {board!r} channel {channel} is outside {MIN_CHANNEL}-{MAX_CHANNEL}"
                    )
                self._switches.append(
                    Switch(board=board, room=room, channel=channel, name=name, host=host)
                )

        # Two switches in one room sharing a name ("bedroom 1 fan" on two
        # different boards) makes every spoken command ambiguous, and the
        # resolver would silently pick one. Catch it at load.
        seen_labels: dict[str, str] = {}
        for switch in self._switches:
            key = switch.label.lower()
            if key in seen_labels:
                raise DeviceConfigError(
                    f"boards {seen_labels[key]!r} and {switch.board!r} both define "
                    f"{switch.label!r} - rename one, or nobody can ask for it by voice"
                )
            seen_labels[key] = switch.board

        for room, spec in (acs or {}).items():
            host = spec.get("host")
            if not host:
                raise DeviceConfigError(f"ac in room {room!r} has no host")
            self._acs.append(AirConditioner(room=room, host=host))

    # -- construction -----------------------------------------------------
    @classmethod
    def load(cls, path: str = CONFIG_PATH) -> "DeviceRegistry":
        if not os.path.exists(path):
            raise DeviceConfigError(
                f"{path} not found. Copy citra_devices.example.json to it and "
                "fill in one block per board."
            )
        with open(path, encoding="utf-8") as fh:
            try:
                raw = json.load(fh)
            except json.JSONDecodeError as exc:
                raise DeviceConfigError(f"{path} is not valid JSON: {exc}") from exc
        return cls(raw.get("boards") or {}, raw.get("acs") or {})

    # -- lookups ----------------------------------------------------------
    @property
    def boards(self) -> list[str]:
        return list(self._board_hosts)

    @property
    def rooms(self) -> list[str]:
        seen = []
        for room in self._board_rooms.values():
            if room not in seen:
                seen.append(room)
        return seen

    @property
    def switches(self) -> list[Switch]:
        return list(self._switches)

    @property
    def air_conditioners(self) -> list[AirConditioner]:
        return list(self._acs)

    def host_for_board(self, board: str) -> str | None:
        return self._board_hosts.get(board)

    def boards_in(self, room: str) -> list[str]:
        return [b for b, r in self._board_rooms.items() if r == room]

    def switches_in(self, room: str) -> list[Switch]:
        """
        Every switch in a room, across ALL its boards - what "turn off the
        hall" means when the hall has three separate switchboards.
        """
        return [s for s in self._switches if s.room == room]

    def switches_on(self, board: str) -> list[Switch]:
        return [s for s in self._switches if s.board == board]

    def ac_in(self, room: str) -> AirConditioner | None:
        for ac in self._acs:
            if ac.room == room:
                return ac
        return None

    # -- resolution -------------------------------------------------------
    def find_switch(self, spoken: str) -> Switch | None:
        """
        Resolve free text like "bedroom 1 fan" or "kitchen light" to one
        switch, or None if nothing is close enough.

        Exact-substring first, fuzzy second. The substring pass exists so
        that an unambiguous phrase never depends on a similarity score:
        "kitchen light" should resolve because it IS the label, not
        because it scored 0.94 against it. Fuzzy matching only handles
        what substring misses - word order, a dropped syllable, a plural.
        """
        text = " ".join(spoken.lower().split())
        if not text:
            return None

        exact = [s for s in self._switches if s.label.lower() == text]
        if len(exact) == 1:
            return exact[0]

        contained = [s for s in self._switches if s.label.lower() in text]
        if len(contained) == 1:
            return contained[0]
        if len(contained) > 1:
            # "hall light" inside "hall main light" - prefer the longest
            # label, which is the most specific thing the user could mean.
            return max(contained, key=lambda s: len(s.label))

        best, best_score = None, 0.0
        for switch in self._switches:
            score = difflib.SequenceMatcher(None, text, switch.label.lower()).ratio()
            if score > best_score:
                best, best_score = switch, score
        return best if best_score >= NAME_MATCH_THRESHOLD else None

    def find_room(self, spoken: str) -> str | None:
        """Resolve free text to a room name, for whole-room commands."""
        text = " ".join(spoken.lower().split())
        if not text:
            return None
        for room in self.rooms:
            if room.lower() in text:
                return room
        best, best_score = None, 0.0
        for room in self.rooms:
            score = difflib.SequenceMatcher(None, text, room.lower()).ratio()
            if score > best_score:
                best, best_score = room, score
        return best if best_score >= NAME_MATCH_THRESHOLD else None

    def describe(self) -> str:
        """One line per board, grouped by room, for startup logging."""
        lines = []
        for room in self.rooms:
            for board in self.boards_in(room):
                names = ", ".join(
                    f"{s.channel}:{s.name}"
                    for s in sorted(self.switches_on(board), key=lambda s: s.channel)
                )
                lines.append(f"  {board:<16} {self._board_hosts[board]:<15} {names}")
            ac = self.ac_in(room)
            if ac:
                lines.append(f"  {room + ' AC':<16} {ac.host:<15} (infrared)")
        return "\n".join(lines)
