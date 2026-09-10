"""
Citra's own phonebook: a name Abhay says, mapped to a number to dial.

WHY THIS EXISTS. Calling by WhatsApp contact name is guesswork. Searching
"maa" returns "Anupama Maam" and "Seema Vishesh Maam" ABOVE "Maa";
"karma wali" failed outright because the real contact carries an emoji.
Every one of those is a chance to ring the wrong person, and the cost of
that is somebody's phone going off at six in the morning.

A number is not guesswork. So Abhay names people here once - including
what he actually CALLS them, which is not what WhatsApp has them saved as
("Shruti", not "Karma Wali") - and Citra dials digits from then on.

THE FILE IS THE INTERFACE. citra_contacts.json is plain, hand-editable
JSON, same as citra_devices.json. Nothing here needs a database, and a
file he can open and read is a file he can fix at 2am without me.

MATCHING IS DELIBERATELY STRICT. A contact must be matched confidently
AND unambiguously, or Citra refuses and says why. Refusing to dial is a
minor annoyance; dialling the wrong person is not.
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
CONTACTS_PATH = os.path.join(HERE, "citra_contacts.json")

# Same bar as citra_whatsapp uses for contact names, and for the same
# reason: a misheard light is a shrug, a misheard person is a phone call.
MATCH_THRESHOLD = 0.82
AMBIGUITY_MARGIN = 0.08

# Indian mobile numbers are 10 digits; with the country code, 12. Anything
# shorter than 7 digits is not a phone number, it is a typo.
MIN_DIGITS = 7
DEFAULT_COUNTRY_CODE = "+91"


class ContactError(ValueError):
    """A contact could not be added - the message explains why."""


def normalise_number(raw: str) -> str:
    """
    Turn whatever was typed into something WhatsApp's keypad accepts.

    Adds the country code when it is missing, because a bare 10-digit
    number resolved to nothing on the keypad and the call silently
    became "would call '9876543210' on 9876543210" - a dial with no
    contact behind it. With +91 the same number resolved to 'Maa'.
    """
    cleaned = re.sub(r"[^\d+]", "", raw or "")
    if not cleaned:
        raise ContactError("that isn't a phone number")

    plus = cleaned.startswith("+")
    digits = re.sub(r"\D", "", cleaned)
    if len(digits) < MIN_DIGITS:
        raise ContactError(f"{raw!r} is too short to be a phone number")

    if plus:
        return "+" + digits
    if len(digits) == 10:               # bare Indian mobile
        return DEFAULT_COUNTRY_CODE + digits
    if digits.startswith("91") and len(digits) == 12:
        return "+" + digits
    if digits.startswith("0") and len(digits) == 11:
        return DEFAULT_COUNTRY_CODE + digits[1:]
    return "+" + digits


def _score(spoken: str, name: str) -> float:
    """
    How well `spoken` identifies `name`, 0..1.

    Coverage-scaled on purpose. A plain substring test scored "maa"
    at 0.92 against "Seema Vishesh Maam" - because "maa" really does
    appear inside "Maam" - which is exactly how the wrong person gets
    called. Matching whole words first, and scaling by how much of the
    name the spoken text actually accounts for, kills that.
    """
    import difflib

    spoken = re.sub(r"\s+", " ", (spoken or "").strip().lower())
    target = re.sub(r"\s+", " ", (name or "").strip().lower())
    if not spoken or not target:
        return 0.0
    if spoken == target:
        return 1.0

    spoken_words = spoken.split()
    target_words = target.split()

    # Whole-word containment: "shruti" in "shruti sharma".
    if all(word in target_words for word in spoken_words):
        coverage = len(spoken_words) / len(target_words)
        return 0.90 + 0.09 * coverage

    # First-name match: "shruti" for "Shruti Sharma".
    if target_words and spoken_words and spoken_words[0] == target_words[0]:
        return 0.88

    ratio = difflib.SequenceMatcher(None, spoken, target).ratio()
    # Substring, but scaled by how much of the name it explains, so a
    # short fragment inside a long name scores low rather than high.
    if spoken in target:
        ratio = max(ratio, 0.70 + 0.25 * (len(spoken) / len(target)))
    return ratio


@dataclass
class Contact:
    name: str
    number: str
    relation: str = ""          # "mother", "girlfriend" - for Citra's tone
    aliases: List[str] = field(default_factory=list)
    note: str = ""

    def all_names(self) -> List[str]:
        return [self.name] + list(self.aliases)

    def to_json(self) -> dict:
        out = {"name": self.name, "number": self.number}
        if self.relation:
            out["relation"] = self.relation
        if self.aliases:
            out["aliases"] = self.aliases
        if self.note:
            out["note"] = self.note
        return out


@dataclass
class Match:
    ok: bool
    reason: str
    contact: Optional[Contact] = None
    confidence: float = 0.0


class ContactBook:
    """
    The phonebook. Loads lazily, saves atomically, and is safe to use
    from the UI server's threads.
    """

    def __init__(self, path: str = CONTACTS_PATH):
        self.path = path
        self._lock = threading.RLock()
        self._contacts: Dict[str, Contact] = {}
        self._loaded = False

    # -- storage --------------------------------------------------------
    def load(self, force: bool = False) -> None:
        with self._lock:
            if self._loaded and not force:
                return
            self._contacts = {}
            self._loaded = True
            if not os.path.exists(self.path):
                return
            try:
                with open(self.path, encoding="utf-8") as handle:
                    raw = json.load(handle)
            except Exception:
                # A corrupt phonebook must not take Citra down. An empty
                # book refuses every call, which is the safe failure.
                return
            for entry in raw.get("contacts", []):
                try:
                    contact = Contact(
                        name=str(entry["name"]).strip(),
                        number=normalise_number(str(entry["number"])),
                        relation=str(entry.get("relation", "")).strip(),
                        aliases=[str(a).strip() for a in entry.get("aliases", [])],
                        note=str(entry.get("note", "")).strip(),
                    )
                except Exception:
                    continue
                if contact.name:
                    self._contacts[contact.name.lower()] = contact

    def save(self) -> None:
        """Atomic write - a half-written phonebook is worse than none."""
        with self._lock:
            payload = {
                "_comment": "Citra's phonebook. Names Abhay actually says, "
                            "mapped to numbers. Safe to edit by hand.",
                "contacts": [c.to_json() for c in sorted(
                    self._contacts.values(), key=lambda c: c.name.lower())],
            }
            temp = self.path + ".tmp"
            with open(temp, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
            os.replace(temp, self.path)

    # -- editing --------------------------------------------------------
    def add(self, name: str, number: str, relation: str = "",
            aliases: Optional[List[str]] = None, note: str = "",
            overwrite: bool = False) -> Contact:
        self.load()
        name = (name or "").strip()
        if not name:
            raise ContactError("a contact needs a name")

        number = normalise_number(number)
        aliases = [a.strip() for a in (aliases or []) if a.strip()]

        with self._lock:
            key = name.lower()
            if key in self._contacts and not overwrite:
                raise ContactError(f"{name!r} is already saved - pass "
                                   f"overwrite to replace them")

            # Two people answering to the same name is how the wrong
            # person gets rung, so it is refused at the point of entry
            # rather than discovered at dialling time.
            for existing in self._contacts.values():
                if existing.name.lower() == key:
                    continue
                clash = {n.lower() for n in existing.all_names()} & \
                        {n.lower() for n in [name] + aliases}
                if clash:
                    raise ContactError(
                        f"{sorted(clash)[0]!r} already points at "
                        f"{existing.name!r}")
                if existing.number == number:
                    raise ContactError(f"that number is already saved as "
                                       f"{existing.name!r}")

            contact = Contact(name=name, number=number, relation=relation,
                              aliases=aliases, note=note)
            self._contacts[key] = contact
            self.save()
            return contact

    def remove(self, name: str) -> bool:
        self.load()
        with self._lock:
            key = (name or "").strip().lower()
            if key not in self._contacts:
                return False
            del self._contacts[key]
            self.save()
            return True

    def all(self) -> List[Contact]:
        self.load()
        with self._lock:
            return sorted(self._contacts.values(), key=lambda c: c.name.lower())

    # -- lookup ---------------------------------------------------------
    def resolve(self, spoken: str) -> Match:
        """
        Who did Abhay mean? Refuses on a weak or ambiguous match.
        """
        self.load()
        spoken = (spoken or "").strip()
        if not spoken:
            return Match(False, "no name given")

        # A number said out loud is not a name - dial it directly.
        if re.fullmatch(r"[\d\s+\-()]{7,}", spoken):
            try:
                number = normalise_number(spoken)
            except ContactError as error:
                return Match(False, str(error))
            return Match(True, "dialling the number as given",
                         Contact(name=number, number=number), 1.0)

        with self._lock:
            scored: List[Tuple[float, Contact]] = []
            for contact in self._contacts.values():
                best = max(_score(spoken, n) for n in contact.all_names())
                scored.append((best, contact))

        if not scored:
            return Match(False, "the phonebook is empty")

        scored.sort(key=lambda pair: pair[0], reverse=True)
        best_score, best_contact = scored[0]

        if best_score < MATCH_THRESHOLD:
            return Match(False,
                         f"closest was {best_contact.name!r} at "
                         f"{best_score:.2f}, below the bar",
                         confidence=best_score)

        if len(scored) > 1:
            runner_up_score, runner_up = scored[1]
            if best_score - runner_up_score < AMBIGUITY_MARGIN - 1e-6:
                return Match(False,
                             f"{best_contact.name!r} and {runner_up.name!r} "
                             f"are too close to tell apart "
                             f"({best_score:.2f} vs {runner_up_score:.2f})",
                             confidence=best_score)

        return Match(True, f"matched {best_contact.name!r}", best_contact,
                     best_score)


_BOOK: Optional[ContactBook] = None


def book() -> ContactBook:
    """The shared phonebook."""
    global _BOOK
    if _BOOK is None:
        _BOOK = ContactBook()
    return _BOOK


if __name__ == "__main__":
    import sys

    b = book()
    if len(sys.argv) < 2:
        print("contacts:")
        for c in b.all():
            extra = f"  ({c.relation})" if c.relation else ""
            alias = f"  aka {', '.join(c.aliases)}" if c.aliases else ""
            print(f"  {c.name:<20} {c.number}{extra}{alias}")
        sys.exit(0)

    command = sys.argv[1]
    if command == "add":
        c = b.add(sys.argv[2], sys.argv[3],
                  relation=sys.argv[4] if len(sys.argv) > 4 else "",
                  aliases=sys.argv[5].split(",") if len(sys.argv) > 5 else None,
                  overwrite=True)
        print(f"saved {c.name} -> {c.number}")
    elif command == "remove":
        print("removed" if b.remove(sys.argv[2]) else "not found")
    elif command == "find":
        m = b.resolve(" ".join(sys.argv[2:]))
        print(f"ok={m.ok} {m.reason} "
              f"({m.contact.number if m.contact else '-'})")


# ----------------------------------------------------------------------
# ROUTER TOOLS - the phonebook by voice
# ----------------------------------------------------------------------

def say_number(number: str) -> str:
    """A phone number as it should be SPOKEN, digit by digit."""
    digits = " ".join(ch for ch in number if ch.isdigit())
    return ("plus " + digits) if number.strip().startswith("+") else digits


def _result(ok: bool, message: str, endpoint: str = "contacts"):
    """
    Every controller method the router dispatches MUST return a
    HardwareResult, not a string.

    This is not cosmetic. jarvis_router reads .success and .message off
    whatever a tool returns, so a plain string raised
    "AttributeError: 'str' object has no attribute 'success'" AFTER the
    contact had already been saved - the work happened, then the reply
    blew up with a 500, so from the chat box it looked like saving a
    contact simply did not work.
    """
    from jarvis_hardware_api import HardwareResult

    return HardwareResult(success=ok, endpoint=endpoint, message=message)


class ContactController:
    """
    Lets Abhay manage the phonebook out loud: "save Shruti's number as
    six two eight three...", "who do you have saved?", "forget Vedant".

    Adding is the only one that can do damage, and the damage is a wrong
    number sitting in the book waiting to be dialled - so the reply
    always reads the saved number BACK. Hearing "+919876500000" out loud
    is how a misheard digit gets caught at the moment it is saved,
    rather than at the moment somebody's phone rings.
    """

    def save_contact(self, name: str, number: str, relation: str = ""):
        try:
            contact = book().add(name, number, relation=relation, overwrite=True)
        except ContactError as error:
            return _result(False, f"I didn't save that - {error}.")
        except Exception:
            return _result(False, "Something went wrong saving that contact.")

        # Digits spaced out so they are heard one at a time - "nine
        # eight seven..." rather than "nine hundred and eighty-seven
        # million...". Joining the raw string spelled out "p l u s",
        # which is worse than not reading it back at all.
        spoken = say_number(contact.number)
        return _result(True, f"Saved {contact.name} as {spoken}. "
                             f"Say it back to me if that's wrong.")

    def list_contacts(self):
        people = book().all()
        if not people:
            return _result(True, "There's nobody in your phonebook yet.")
        if len(people) == 1:
            return _result(True, f"Just {people[0].name}.")
        names = [c.name for c in people]
        return _result(True, "You have " + ", ".join(names[:-1]) +
                             " and " + names[-1] + ".")

    def forget_contact(self, name: str):
        match = book().resolve(name)
        if not match.ok or match.contact is None:
            return _result(False, f"I couldn't find {name} in your phonebook.")
        if book().remove(match.contact.name):
            return _result(True, f"Removed {match.contact.name}.")
        return _result(False, f"I couldn't remove {name}.")


CONTACT_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "save_contact",
            "description": (
                "Save somebody's phone number in Abhay's phonebook so Citra can "
                "call them later by name. Use when he says things like 'save "
                "Shruti's number as ...', 'add Maa, her number is ...', "
                "'remember this number for Papa'. This does NOT place a call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "What Abhay calls this person - the name he "
                                       "will use later when asking to ring them, "
                                       "e.g. 'Shruti'. Not their WhatsApp display "
                                       "name.",
                    },
                    "number": {
                        "type": "string",
                        "description": "The phone number, digits only or with a "
                                       "country code. A bare 10-digit Indian mobile "
                                       "is fine; +91 is added automatically.",
                    },
                    "relation": {
                        "type": "string",
                        "description": "Optional - 'mother', 'girlfriend', 'friend'. "
                                       "Only if he says it.",
                    },
                },
                "required": ["name", "number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_contacts",
            "description": (
                "Say who is saved in Abhay's phonebook. Use for 'who do you have "
                "saved', 'what numbers do you know', 'read out my contacts'."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget_contact",
            "description": (
                "Remove somebody from Abhay's phonebook. Use for 'forget Vedant', "
                "'delete Shruti's number', 'remove Maa from your contacts'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string",
                             "description": "Who to remove, as Abhay said it."},
                },
                "required": ["name"],
            },
        },
    },
]
