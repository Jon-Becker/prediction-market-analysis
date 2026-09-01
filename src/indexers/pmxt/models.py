"""Configuration models for the read-only PMXT Router sync."""

from __future__ import annotations

from dataclasses import dataclass

MATCH_RELATIONS = frozenset({"identity", "complement", "subset", "superset", "overlap", "disjoint"})
SORT_ORDERS = frozenset({"volume", "confidence"})


@dataclass(frozen=True)
class PmxtQuery:
    """The deliberately narrow query used by the first PMXT milestone."""

    relation: str = "identity"
    min_confidence: float = 0.80
    min_venues: int = 2
    limit: int = 25
    offset: int = 0
    sort: str = "volume"
    venues: tuple[str, ...] = ("kalshi", "polymarket")
    include_raw_matches: bool = True

    def __post_init__(self) -> None:
        relation = self.relation.strip().lower()
        if relation not in MATCH_RELATIONS:
            raise ValueError(f"Unsupported PMXT relation: {self.relation!r}")
        object.__setattr__(self, "relation", relation)

        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        if self.min_venues < 2:
            raise ValueError("min_venues must be at least 2 for cross-venue candidates")
        if not 1 <= self.limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        if self.offset < 0:
            raise ValueError("offset must be non-negative")

        sort = self.sort.strip().lower()
        if sort not in SORT_ORDERS:
            raise ValueError(f"Unsupported PMXT sort order: {self.sort!r}")
        object.__setattr__(self, "sort", sort)

        venues = tuple(dict.fromkeys(venue.strip().lower() for venue in self.venues if venue.strip()))
        object.__setattr__(self, "venues", venues)

    def as_params(self) -> dict[str, str]:
        """Return documented Router query parameters without credentials."""

        params = {
            "relation": self.relation,
            "minConfidence": f"{self.min_confidence:g}",
            "minVenues": str(self.min_venues),
            "includeRawMatches": str(self.include_raw_matches).lower(),
            "sort": self.sort,
            "limit": str(self.limit),
            "offset": str(self.offset),
        }
        if self.venues:
            params["venues"] = ",".join(self.venues)
        return params

    def as_dict(self) -> dict:
        """Return a JSON-safe query description for the run manifest."""

        return {
            "relation": self.relation,
            "min_confidence": self.min_confidence,
            "min_venues": self.min_venues,
            "limit": self.limit,
            "offset": self.offset,
            "sort": self.sort,
            "venues": list(self.venues),
            "include_raw_matches": self.include_raw_matches,
        }
