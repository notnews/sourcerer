"""The claim-to-source bridge: name resolution and the two typing routes.

These tests pin the behaviours that the 100-article pilot showed to matter, in
particular surname aliasing and the guard against it. Both are easy to break
in a way that raises no error and silently changes the reported match rate.
"""

from __future__ import annotations

import pytest

from sourcerer.sources import SpeakerIndex, classify_claimer

covered = pytest.importorskip("covered", reason="source typing needs sourcerer[taxonomy]")


# CNN's actual convention: full introduction once, surname alone thereafter.
TRANSCRIPT = """\
JOHN BERMAN, CNN ANCHOR: Joining us now is Congressman Ruben Gallego.

REP. RUBEN GALLEGO, (D) ARIZONA: Thanks for having me.

BERMAN: What is your read on the vote?

GALLEGO: There was a conspiracy to stop the transfer of power.
"""


@pytest.fixture
def index() -> SpeakerIndex:
    return SpeakerIndex.from_transcript(TRANSCRIPT)


def test_full_name_resolves_exactly(index: SpeakerIndex) -> None:
    resolved = index.resolve("Ruben Gallego")
    assert resolved.route == "exact"
    assert resolved.role_raw, "the introduction turn's role clause should be kept"


def test_surname_only_turns_resolve_to_the_introduced_speaker() -> None:
    """The model reports the introduced name; most turns carry only a surname.

    Without this tier the guest match rate on the pilot was 81.5% rather than
    94.2%, and the role clause needed for classification was never reached.
    """
    text = "GALLEGO, (D) ARIZONA: There was a conspiracy.\n\nGALLEGO: More here.\n"
    resolved = SpeakerIndex.from_transcript(text).resolve("Ruben Gallego")
    assert resolved.route == "surname"
    assert resolved.name == "gallego"


def test_a_different_first_name_blocks_a_surname_match() -> None:
    """Same surname, different person -- the case that inflates match rates.

    A quoted "Donald Trump" must not bind to an on-air "Mary Trump". This guard
    rejected 99 of 420 candidate surname matches in the pilot.
    """
    text = "MARY TRUMP, AUTHOR: He was always like this.\n"
    resolved = SpeakerIndex.from_transcript(text).resolve("Donald Trump")
    assert resolved.name is None
    assert resolved.route == "none"


def test_an_ambiguous_surname_does_not_resolve() -> None:
    text = (
        "JOHN SMITH, CNN CORRESPONDENT: One.\n\nJANE SMITH, ANALYST: Two.\n"
    )
    assert SpeakerIndex.from_transcript(text).resolve("Adam Smith").name is None


def test_an_unknown_name_does_not_resolve(index: SpeakerIndex) -> None:
    assert index.resolve("Nobody Here").name is None
    assert index.resolve("").name is None


def test_on_air_claimer_is_typed_from_the_transcript_role(index: SpeakerIndex) -> None:
    label, route = classify_claimer("Ruben Gallego", "guest", index)
    assert route in {"exact", "surname"}
    assert label.sector == "government_legislative"


def test_quoted_source_is_typed_from_its_own_description() -> None:
    """A quoted source has no turn to look up, so the string is the role clause."""
    label, route = classify_claimer("The Washington Post", "quoted_source", None)
    assert route == "string"
    assert label.sector == "media"


def test_a_bare_name_with_no_role_stays_unknown() -> None:
    """Typing a person from their name alone needs a registry we do not have.

    Returning unknown keeps these countable instead of guessing a sector.
    """
    label, route = classify_claimer("Olga Smirnova", "quoted_source", None)
    assert label.sector == "unknown"
    assert route == "none"


def test_a_networks_own_analyst_is_typed_as_media() -> None:
    """The gap covered leaves on purpose, which sourcerer has to fill.

    covered excludes CNN staff from the sources it counts, so it never needed
    to sector them; here they appear as guests, and a salaried network analyst
    is a different epistemic object from an outside expert. This rule took
    guest sector coverage from 58.0% to 73.9% on the pilot.
    """
    text = "PETER BERGEN, CNN NATIONAL SECURITY ANALYST: The strike was planned.\n"
    idx = SpeakerIndex.from_transcript(text)
    label, _ = classify_claimer("Peter Bergen", "guest", idx)
    assert label.sector == "media"
    assert label.sector_source == "outlet_badge"
    assert label.epistemic_role == "expert", "covered's own role reading is kept"


def test_an_outlet_word_alone_does_not_make_someone_media() -> None:
    """"TIME" and "VOX" are outlets and also ordinary words.

    Requiring a newsroom job word alongside the outlet is what stops a role
    clause from being read as media just because it contains one.
    """
    from sourcerer.sources import _is_network_staff

    assert _is_network_staff("CNN POLITICAL ANALYST")
    assert not _is_network_staff("FORMER TIME KEEPER")
    assert not _is_network_staff("SECRETARY OF DEFENSE")


def test_newsroom_staff_are_typed_by_what_they_are(index: SpeakerIndex) -> None:
    """An anchor is the newsroom speaking, so the sector is not a finding.

    Routing them through the lexicon returned "unknown" for 95% of anchors,
    because covered has CORRESPONDENT but deliberately not ANCHOR or HOST.
    """
    label, route = classify_claimer("John Berman", "anchor", index)
    assert route == "newsroom"
    assert label.sector == "media"


TRANSCRIPT_WITH_ENTITIES = """\
JOHN BERMAN, CNN ANCHOR: President Joe Biden spoke today. Volodymyr Zelenskyy
responded from Kyiv, and Zelenskyy's chief of staff briefed reporters. Local
officials in Kyiv confirmed it, and a top Russian general disputed it. The
Washington Post reported the exchange.
"""


@pytest.fixture(scope="module")
def entities():
    from covered.ner import load_nlp

    from sourcerer.sources import EntityIndex

    return EntityIndex.from_transcript(TRANSCRIPT_WITH_ENTITIES, load_nlp("en_core_web_sm"))


def test_a_name_resolves_through_the_transcript(entities) -> None:
    """In-context typing, which bare-string NER gets wrong on non-Anglo names.

    Running NER on "Volodymyr Zelenskyy" alone returns nothing from the large
    model; in the sentence it came from it is a person. A measure that fails
    hardest on foreign names is the wrong measure for a foreign-news corpus.
    """
    assert entities.form("Volodymyr Zelenskyy") == "person"
    assert entities.form("President Joe Biden") == "person"


def test_a_relational_description_is_not_the_person_it_mentions(entities) -> None:
    """The false positive that would destroy the unnamed-source count.

    "Zelenskyy's chief of staff" is an unnamed person standing next to a named
    one. Substring matching typed it as Zelenskyy, and "Local officials in
    Kyiv" as the city -- converting the one category that cannot be counted
    into the one that can, which is the opposite of what the bounds are for.
    """
    assert entities.form("Zelenskyy's chief of staff") is None
    assert entities.form("Local officials in Kyiv") is None
    assert entities.form("Peskov's son") is None


def test_a_role_description_does_not_reach_a_nearby_country(entities) -> None:
    """The head noun must appear in whatever matched it.

    A name's head is its surname; a description's head is a common noun no
    entity carries. That is what stops "top Russian general" reaching "Russia".
    """
    assert entities.form("top Russian general") is None


def test_an_unknown_name_is_absent_rather_than_guessed(entities) -> None:
    assert entities.form("Nobody Whatsoever") is None
    assert entities.form("") is None


def test_describe_source_types_and_genders_a_person(entities) -> None:
    from sourcerer.sources import describe_source

    label, _ = classify_claimer("Volodymyr Zelenskyy", "quoted_source", None)
    described = describe_source(label, "Volodymyr Zelenskyy", entities)
    assert described.entity_form == "person"
    assert described.gender == "male"
    assert described.gender_source in {"wikidata", "name_model"}


def test_an_organisation_gets_no_gender(entities) -> None:
    """There is nothing to read, so nothing is read."""
    from sourcerer.sources import describe_source

    label, _ = classify_claimer("The Washington Post", "quoted_source", None)
    described = describe_source(label, "The Washington Post", entities)
    assert described.entity_form == "organization"
    assert described.gender == "unknown"
    assert described.gender_source == "none"


def test_on_air_role_falls_back_to_the_string_when_unmatched(index: SpeakerIndex) -> None:
    label, route = classify_claimer("A Senior U.S. Official", "guest", index)
    assert route in {"string", "none"}
    assert label is not None
