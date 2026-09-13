"""
The phonebook is the one place a fuzzy match has a real cost: a weak or
ambiguous name match rings the wrong person. These tests pin the number
normalisation, the coverage-scaled scoring, and the refuse-when-unsure
behaviour of ContactBook, against a phonebook in a temp directory.
"""
import json

import pytest

import citra_contacts
from citra_contacts import (
    AMBIGUITY_MARGIN,
    MATCH_THRESHOLD,
    ContactBook,
    ContactController,
    ContactError,
    _score,
    normalise_number,
    say_number,
)


# --------------------------------------------------------------------------
# normalise_number
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("9876543210", "+919876543210"),          # bare Indian mobile
        ("98765 43210", "+919876543210"),         # spoken with a pause
        ("+91 98765-43210", "+919876543210"),     # already has the code
        ("919876543210", "+919876543210"),        # code without the plus
        ("09876543210", "+919876543210"),         # trunk-prefix form
        ("+1 (415) 555-0100", "+14155550100"),    # a non-Indian number
        ("00 44 20 7946 0958", "+00442079460958"),  # unknown shape: kept
    ],
)
def test_normalise_number_produces_a_keypad_dialable_string(raw, expected):
    assert normalise_number(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "call mum", "+", "12345", "1-2-3"])
def test_normalise_number_rejects_things_that_are_not_numbers(raw):
    with pytest.raises(ContactError):
        normalise_number(raw)


def test_say_number_reads_digits_one_at_a_time():
    assert say_number("+919876543210") == "plus 9 1 9 8 7 6 5 4 3 2 1 0"
    assert say_number("415") == "4 1 5"


# --------------------------------------------------------------------------
# _score
# --------------------------------------------------------------------------
def test_exact_name_scores_one():
    assert _score("Shruti Sharma", "shruti sharma") == 1.0


def test_first_name_alone_is_a_strong_match():
    assert _score("shruti", "Shruti Sharma") >= MATCH_THRESHOLD


def test_a_short_fragment_inside_a_long_name_does_not_pass():
    # The bug this scoring exists for: "maa" is inside "Maam", and a plain
    # substring test rated it 0.92 - which is how the wrong person gets
    # called. It must stay below the bar.
    assert _score("maa", "Seema Vishesh Maam") < MATCH_THRESHOLD


def test_whole_word_coverage_beats_partial_coverage():
    assert _score("shruti sharma", "Shruti Sharma") > _score("shruti", "Shruti Sharma")


def test_empty_inputs_score_zero():
    assert _score("", "anyone") == 0.0
    assert _score("anyone", "") == 0.0


# --------------------------------------------------------------------------
# ContactBook
# --------------------------------------------------------------------------
@pytest.fixture
def book(tmp_path) -> ContactBook:
    return ContactBook(path=str(tmp_path / "phonebook.json"))


def test_add_then_resolve_round_trips_through_disk(book, tmp_path):
    book.add("Shruti", "9876543210", relation="friend", aliases=["shru"])

    reloaded = ContactBook(path=book.path)
    match = reloaded.resolve("shruti")

    assert match.ok
    assert match.contact.number == "+919876543210"
    assert match.contact.relation == "friend"
    # Written atomically: no leftover temp file next to the real one.
    assert not (tmp_path / "phonebook.json.tmp").exists()


def test_aliases_resolve_too(book):
    book.add("Mother", "9876543210", aliases=["maa", "mum"])
    assert book.resolve("maa").contact.name == "Mother"
    assert book.resolve("mum").contact.name == "Mother"


def test_adding_the_same_name_twice_needs_overwrite(book):
    book.add("Vedant", "9876543210")
    with pytest.raises(ContactError, match="already saved"):
        book.add("Vedant", "9876543211")

    book.add("Vedant", "9876543211", overwrite=True)
    assert book.resolve("vedant").contact.number == "+919876543211"


def test_two_people_cannot_share_a_name_or_alias(book):
    book.add("Rahul", "9876543210", aliases=["bhai"])
    with pytest.raises(ContactError, match="already points at"):
        book.add("Rohan", "9876543299", aliases=["bhai"])


def test_the_same_number_cannot_be_saved_under_two_names(book):
    book.add("Rahul", "9876543210")
    with pytest.raises(ContactError, match="already saved as"):
        book.add("Someone Else", "+91 98765 43210")


def test_a_contact_needs_a_name(book):
    with pytest.raises(ContactError, match="needs a name"):
        book.add("   ", "9876543210")


def test_resolve_refuses_a_weak_match(book):
    book.add("Shruti Sharma", "9876543210")
    match = book.resolve("kamlesh")

    assert not match.ok
    assert "below the bar" in match.reason
    assert match.confidence < MATCH_THRESHOLD


def test_resolve_refuses_an_ambiguous_match(book):
    book.add("Priya Nair", "9876543210")
    book.add("Priya Menon", "9876543211")
    match = book.resolve("priya")

    assert not match.ok
    assert "too close" in match.reason


def test_resolve_prefers_a_clear_winner_over_a_close_second(book):
    book.add("Priya Nair", "9876543210")
    book.add("Preeti", "9876543211")
    match = book.resolve("priya nair")

    assert match.ok
    assert match.contact.name == "Priya Nair"
    assert match.confidence - AMBIGUITY_MARGIN > 0


def test_a_spoken_number_is_dialled_directly(book):
    match = book.resolve("98765 43210")
    assert match.ok
    assert match.contact.number == "+919876543210"
    assert match.confidence == 1.0


def test_an_empty_phonebook_refuses_every_name(book):
    match = book.resolve("anyone")
    assert not match.ok
    assert "empty" in match.reason


def test_blank_input_is_refused(book):
    assert not book.resolve("   ").ok


def test_a_corrupt_phonebook_loads_as_empty_not_a_crash(tmp_path):
    path = tmp_path / "phonebook.json"
    path.write_text("{ this is not json", encoding="utf-8")

    book = ContactBook(path=str(path))
    assert book.all() == []


def test_entries_with_bad_numbers_are_skipped_on_load(tmp_path):
    path = tmp_path / "phonebook.json"
    path.write_text(json.dumps({"contacts": [
        {"name": "Good", "number": "9876543210"},
        {"name": "Bad", "number": "not a number"},
    ]}), encoding="utf-8")

    names = [c.name for c in ContactBook(path=str(path)).all()]
    assert names == ["Good"]


def test_remove_returns_whether_anything_was_removed(book):
    book.add("Vedant", "9876543210")
    assert book.remove("vedant") is True
    assert book.remove("vedant") is False
    assert book.all() == []


# --------------------------------------------------------------------------
# ContactController (the voice-facing tools)
# --------------------------------------------------------------------------
@pytest.fixture
def controller(book, monkeypatch) -> ContactController:
    monkeypatch.setattr(citra_contacts, "_BOOK", book)
    return ContactController()


def test_save_contact_reads_the_number_back(controller):
    result = controller.save_contact("Shruti", "9876543210")

    assert result.success is True
    assert "plus 9 1 9 8 7 6 5 4 3 2 1 0" in result.message


def test_save_contact_reports_a_bad_number_without_raising(controller):
    result = controller.save_contact("Shruti", "banana")

    assert result.success is False
    assert "didn't save" in result.message


def test_list_contacts_reads_naturally(controller):
    assert "nobody" in controller.list_contacts().message
    controller.save_contact("A", "9876543210")
    assert controller.list_contacts().message == "Just A."
    controller.save_contact("B", "9876543211")
    controller.save_contact("C", "9876543212")
    assert controller.list_contacts().message == "You have A, B and C."


def test_forget_contact_uses_the_same_fuzzy_resolution(controller):
    controller.save_contact("Vedant Kumar", "9876543210")

    assert controller.forget_contact("vedant").success is True
    assert controller.forget_contact("vedant").success is False
