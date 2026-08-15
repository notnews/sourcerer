"""Type the source of a claim by joining it to ``covered``'s taxonomy.

sourcerer and covered answer different questions about the same transcripts.
covered asks who is given a microphone; sourcerer asks how well supported each
claim is. The split of labour here follows from that: the LLM decides what is a
claim and whether it is supported -- the part only a language model can do --
and ``covered.taxonomy.classify`` decides what kind of source the claimer is,
deterministically and auditably, with no API key involved.

That is not merely tidier. The old prompt did both jobs at once and told the
model that for an attributed claim "the evidence IS the attribution", yet 73.1%
of guest claims still came back ``unsupported``. Asking one prompt to classify a
source and judge support put those two answers in tension. Moving source typing
out of the model removes the tension at the root.

The join is by speaker name within a transcript, and its one non-obvious part is
surname aliasing. CNN introduces a guest in full once and labels every later
turn with the surname alone::

    REP. RUBEN GALLEGO, (D) ARIZONA: ...
    GALLEGO: ...

The model reports the full name it read in the introduction, so an exact match
on the speaker label misses most of that guest's turns -- and the role clause
that ``classify`` needs sits on the introduction turn, not the surname ones.
Resolving surnames to the introduced speaker lifted the guest match rate from
81.5% to 94.2% on the 100-article pilot.

Surname matching can also invent matches, so it is guarded: a surname resolves
only when exactly one speaker in that transcript carries it, and never when both
sides have a first name and the first names disagree. Without that second rule
a quoted "Donald Trump" binds to an on-air "Mary Trump". The guard rejected 99
of 420 candidate surname matches in the pilot, which is why the honest overall
rate is 89.4% rather than the 92.2% an unguarded matcher reports.
"""

from __future__ import annotations

import functools
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from covered.taxonomy import SourceLabel

# Staff of the news organisation. Their sector is not a finding, it is what they
# are: an anchor speaking is CNN speaking. Sending them through the taxonomy
# invites a wrong answer for no gain -- covered's lexicon has CORRESPONDENT but
# deliberately not ANCHOR or HOST, because covered excludes staff from the
# sources it counts, so 95% of anchors came back "unknown". For these claims the
# open question is how the newsroom knows the thing, which is the model's job,
# not whose voice it is.
NEWSROOM_ROLES = frozenset({"anchor", "reporter"})

# Guests speak on air, so a turn exists whose role clause can be classified.
# A quoted_source is by definition being quoted *by* someone else and usually
# has no turn at all; that is the normal case for them, not a join failure.
ON_AIR_ROLES = frozenset({"guest"})

_MISSING_COVERED = (
    "Source typing needs the covered package: pip install 'sourcerer[taxonomy]'. "
    "Claim extraction and support classification do not need it."
)


def _covered():
    """Import covered on demand, with an actionable message when it is absent."""
    try:
        from covered import entities, speakers, taxonomy
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(_MISSING_COVERED) from exc
    return entities, speakers, taxonomy


# Job words that make someone a paid voice of a news organisation rather than an
# independent source. Requiring one of these alongside an outlet name keeps
# "TIME" or "VOX" from matching a role clause that merely contains the word.
_MEDIA_JOBS = frozenset(
    {
        "analyst",
        "anchor",
        "columnist",
        "commentator",
        "contributor",
        "correspondent",
        "editor",
        "host",
        "journalist",
        "pundit",
        "reporter",
    }
)

# Networks whose own staff appear as guests here. covered's outlet list omits
# CNN deliberately -- its corpus *is* CNN, so CNN staff are identified as staff
# and excluded from the sources it counts. sourcerer must type them instead: a
# network's salaried analyst introduced as a guest is a different epistemic
# object from an outside expert, and telling those apart is the point.
_HOME_NETWORKS = frozenset({"cnn"})


@functools.lru_cache(maxsize=1)
def _outlet_names() -> frozenset[str]:
    """Outlet vocabulary, reusing covered's curated list."""
    from covered.taxonomy import REFERENCE

    path = REFERENCE / "news_outlets.txt"
    listed = {
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    return frozenset(listed | _HOME_NETWORKS)


@functools.lru_cache(maxsize=2048)
def _is_network_staff(role_clean: str) -> bool:
    """Whether a role clause names both a news outlet and a newsroom job.

    Args:
        role_clean: Role clause as CNN wrote it, e.g. ``CNN GLOBAL AFFAIRS
            ANALYST``.

    Returns:
        True when the clause identifies a paid voice of a news organisation.
    """
    low = role_clean.lower()

    def word(term: str) -> bool:
        return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", low) is not None

    return any(word(j) for j in _MEDIA_JOBS) and any(word(o) for o in _outlet_names())


@dataclass(frozen=True)
class Resolution:
    """Outcome of matching one claimer string to a speaker in one transcript.

    Attributes:
        name: Normalised speaker name that was matched, or ``None`` if unmatched.
        role_raw: Role clause from that speaker's introduction turn. Empty when
            the speaker was matched but never introduced with a role.
        route: How the match was made -- ``"exact"``, ``"surname"``, or
            ``"none"``. Recorded so the surname tier can be audited or switched
            off without re-deriving which matches depended on it.
    """

    name: str | None
    role_raw: str
    route: str


class SpeakerIndex:
    """Speakers of one transcript, queryable by the name a model reports.

    Attributes:
        roles: Normalised speaker name to the role clause first seen for them.
    """

    def __init__(self, roles: dict[str, str]) -> None:
        """Build an index over already-normalised speaker names.

        Args:
            roles: Normalised speaker name to role clause (empty when unknown).
        """
        self.roles = roles
        self._by_surname: dict[str, set[str]] = defaultdict(set)
        for name in roles:
            self._by_surname[name.split()[-1]].add(name)

    @classmethod
    def from_transcript(cls, text: str) -> SpeakerIndex:
        """Parse ``text`` into speaker turns and index them.

        Args:
            text: Full transcript text of a single programme.

        Returns:
            An index over every speaker that appears in the transcript.
        """
        entities, speakers, _ = _covered()
        roles: dict[str, str] = {}
        for turn in speakers.parse_turns(text):
            # parse_turns already folds surname-only turns onto the introduced
            # speaker, but its name_norm keeps the honorific ("rep. ruben
            # gallego") while a model reports the bare name. Normalising again
            # strips the title and makes the two sides comparable; skipping this
            # step is what held the guest match rate down to 81.5%.
            name = entities.normalize_name(turn.name_norm or turn.speaker_raw or "")
            if not name:
                continue
            # A speaker's role clause appears on their introduction turn; later
            # surname-only turns carry none. Keep the first non-empty one.
            if turn.role_raw and not roles.get(name):
                roles[name] = turn.role_raw
            else:
                roles.setdefault(name, "")
        return cls(roles)

    def resolve(self, claimer: str) -> Resolution:
        """Match a model-reported claimer name to a speaker in this transcript.

        Args:
            claimer: Name as the model reported it, e.g. ``"Ruben Gallego"``.

        Returns:
            The resolution, with ``name`` set to ``None`` when no speaker
            matches or when a surname match was rejected as ambiguous.
        """
        entities, _, _ = _covered()
        name = entities.normalize_name(claimer or "")
        if not name:
            return Resolution(None, "", "none")
        if name in self.roles:
            return Resolution(name, self.roles[name], "exact")

        candidates = self._by_surname.get(name.split()[-1], set())
        if len(candidates) != 1:
            return Resolution(None, "", "none")
        candidate = next(iter(candidates))
        # Same surname, different first name is a different person: a quoted
        # "Donald Trump" must not bind to an on-air "Mary Trump".
        if " " in name and " " in candidate and name.split()[0] != candidate.split()[0]:
            return Resolution(None, "", "none")
        return Resolution(candidate, self.roles[candidate], "surname")


def _fill_network_staff(label: SourceLabel, role_raw: str) -> SourceLabel:
    """Type a network's own on-air staff as media when no lexicon rule fired.

    covered leaves ``CNN GLOBAL AFFAIRS ANALYST`` unsectored because it treats
    CNN people as staff and excludes them from the sources it counts. Here they
    are guests making claims, and the fact that the network pays them is the
    interesting part, so the gap has to be filled rather than inherited.

    Sector only. The epistemic role covered derived (``CNN CONTRIBUTOR`` to
    commentator, ``... ANALYST`` to expert) is passed through untouched, and
    where covered could not resolve one it stays unresolved rather than being
    defaulted -- an anchor reading copy is not obviously a commentator, and an
    invented value would be indistinguishable from a measured one downstream.

    Args:
        label: Label as covered classified it.
        role_raw: Role clause the label came from.

    Returns:
        The label, with sector filled in where the clause names a newsroom job
        at a named outlet.
    """
    if label.sector != "unknown" or not role_raw or not _is_network_staff(role_raw):
        return label
    return replace(label, sector="media", sector_source="outlet_badge")


def classify_claimer(
    claimer: str,
    claimer_role: str,
    index: SpeakerIndex | None = None,
) -> tuple[SourceLabel, str]:
    """Type the source of a claim on covered's crossed axes.

    Two routes, chosen by whether the claimer spoke on air:

    * anchor, reporter, guest -- look the speaker up in ``index`` and classify
      the role clause CNN attached to their introduction.
    * quoted_source -- there is no turn to look up, so classify the claimer
      string itself. ``classify`` takes a role clause, which is exactly what a
      description like ``"a senior U.S. official"`` or ``"The Washington Post"``
      is. A bare personal name carries no role and stays unknown rather than
      being guessed at.

    Args:
        claimer: Name or description of who made the claim.
        claimer_role: One of ``anchor``, ``reporter``, ``guest``,
            ``quoted_source``.
        index: Speakers of the transcript the claim came from. Optional so that
            callers without the transcript still get the quoted_source route.

    Returns:
        The crossed-axis label, and the route used to obtain it -- ``"exact"``,
        ``"surname"``, ``"string"`` (classified the claimer text itself), or
        ``"none"``. The route is returned rather than logged because the match
        rate is a reportable property of the pipeline, not a diagnostic.
    """
    _, _, taxonomy = _covered()
    if claimer_role in NEWSROOM_ROLES:
        role_raw = index.resolve(claimer).role_raw if index is not None else ""
        label = taxonomy.classify(role_raw or None, claimer)
        if label.sector == "unknown":
            # Keep a richer reading when the role clause has one -- a "CNN HOST
            # AND SENIOR LEGAL ANALYST" really is judicial_legal -- and fall
            # back to media only where the lexicon has nothing, recording that
            # the sector came from the speaker's job and not from a rule.
            label = replace(label, sector="media", sector_source="newsroom_role")
        return label, "newsroom"
    if claimer_role in ON_AIR_ROLES and index is not None:
        resolved = index.resolve(claimer)
        if resolved.name is not None:
            label = taxonomy.classify(resolved.role_raw or None, claimer)
            return _fill_network_staff(label, resolved.role_raw), resolved.route
    label = taxonomy.classify(claimer or None, None)
    # classify signals "no rule fired" as sector unknown with an epistemic role
    # of unknown or unresolved. Calling that a successful "string" route would
    # report a typing rate that counts every failure as a success.
    typed = label.sector != "unknown" or label.epistemic_role not in {
        "unknown",
        "unresolved",
    }
    return label, "string" if typed else "none"
