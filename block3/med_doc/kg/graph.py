"""Clinical Knowledge Graph engine for test lookups, profile expansions, and tube priors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from med_doc.kg.schemas import CatalogueItem, RankedCandidate, ValidationResult
from med_doc.kg.textutil import normalize_text, token_similarity
from med_doc.paths import DEFAULT_KG


def _tier_from_score(score: float) -> int:
    if score >= 0.75:
        return 1
    if score >= 0.40:
        return 2
    return 3


class KnowledgeGraph:
    """Frozen Clinical Knowledge Graph providing deterministic lookups and validation."""

    _instances: dict[str, KnowledgeGraph] = {}

    def __init__(self, data: dict[str, Any]):
        self.raw_data = data
        self.version = data.get("version", "v1")
        self.disclaimer = data.get("disclaimer", "Clinical guidance only. Not diagnostic advice.")

        # Catalogue items
        self.catalogue: dict[str, CatalogueItem] = {}
        self._label_to_id: dict[str, str] = {}
        for row in data.get("catalogue", []):
            item = CatalogueItem(**row)
            self.catalogue[item.field_id] = item
            
            # Map field_id directly
            self._label_to_id[normalize_text(item.field_id)] = item.field_id
            
            # Map full label
            self._label_to_id[normalize_text(item.label)] = item.field_id
            
            # Sub-components from slash or parens in label (e.g., 'ALT / SGPT', 'RPR (VDRL)')
            for part in item.label.replace("(", "/").replace(")", "/").split("/"):
                part_clean = normalize_text(part)
                if part_clean and part_clean not in self._label_to_id:
                    self._label_to_id[part_clean] = item.field_id
            
            for alias in item.aliases:
                self._label_to_id[normalize_text(alias)] = item.field_id
                for part in alias.replace("(", "/").replace(")", "/").split("/"):
                    part_clean = normalize_text(part)
                    if part_clean and part_clean not in self._label_to_id:
                        self._label_to_id[part_clean] = item.field_id

        # Profile bundles
        self.profile_bundles: dict[str, list[str]] = data.get("profile_bundles", {})

        # Test to tube mapping
        self.test_to_tube: dict[str, list[str]] = data.get("test_to_tube", {})
        self.tube_field_to_tube: dict[str, str] = data.get("tube_field_to_tube", {})

        # Abbreviations & Misspellings
        self.abbreviations: dict[str, str] = {
            normalize_text(k): normalize_text(v) for k, v in data.get("abbreviations", {}).items()
        }
        self.misspellings: dict[str, str] = {
            normalize_text(k): normalize_text(v) for k, v in data.get("misspellings", {}).items()
        }

        # Write-ins & tube rules
        self.write_ins: set[str] = {normalize_text(x) for x in data.get("write_ins", [])}
        self.write_in_tubes: dict[str, list[str]] = {
            normalize_text(k): list(v) for k, v in data.get("write_in_tubes", {}).items()
        }

        # Lexicons
        self.lexicons: dict[str, list[str]] = {
            k: [str(x) for x in v] for k, v in data.get("lexicons", {}).items()
        }

    @classmethod
    def load(cls, path: str | Path | None = None) -> KnowledgeGraph:
        """Load and cache a KnowledgeGraph instance from a JSON file."""
        p = Path(path) if path else DEFAULT_KG
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in cls._instances:
            if not p.exists():
                raise FileNotFoundError(f"Knowledge Graph file not found: {p}")
            data = json.loads(p.read_text(encoding="utf-8"))
            cls._instances[key] = cls(data)
        return cls._instances[key]

    def get_item(self, field_id: str) -> CatalogueItem | None:
        """Get catalogue item metadata by field ID."""
        return self.catalogue.get(field_id)

    def expand_profile(self, profile_id: str) -> list[str]:
        """Return list of individual test field IDs included in a profile bundle."""
        return list(self.profile_bundles.get(profile_id, []))

    def implied_tests(self, ticked_ids: list[str]) -> set[str]:
        """Return all test IDs implied by the currently ticked profiles and tests."""
        implied: set[str] = set()
        for fid in ticked_ids:
            if fid in self.profile_bundles:
                implied.update(self.profile_bundles[fid])
            item = self.catalogue.get(fid)
            if item and item.components:
                implied.update(item.components)
        return implied

    def calculate_expected_tubes(self, ticked_ids: list[str]) -> dict[str, int]:
        """Calculate required specimen tubes based on ordered tests and profiles."""
        all_tests = set(ticked_ids)
        all_tests.update(self.implied_tests(ticked_ids))

        tube_types: set[str] = set()
        for fid in all_tests:
            # Check test_to_tube
            if fid in self.test_to_tube:
                tube_types.update(self.test_to_tube[fid])
            # Check catalogue entry
            item = self.catalogue.get(fid)
            if item and item.tubes:
                tube_types.update(item.tubes)

        # Count 1 tube per required unique specimen container
        counts: dict[str, int] = {}
        for t in tube_types:
            counts[t] = 1

        # Glucose tolerance / OGTT requires 2-3 Fl tubes if multiple points
        if "profile_ogtt" in ticked_ids or "ogtt" in ticked_ids:
            counts["Fl"] = max(counts.get("Fl", 0), 2)

        return counts

    def normalize_term(self, text: str) -> str:
        """Resolve misspellings and expand abbreviations."""
        n = normalize_text(text)
        if n in self.misspellings:
            n = self.misspellings[n]
        if n in self.abbreviations:
            n = self.abbreviations[n]
        return n

    def resolve_alias(self, text: str) -> str | None:
        """Find the canonical field_id for a given label, abbreviation, or alias."""
        n = self.normalize_term(text)
        if n in self._label_to_id:
            return self._label_to_id[n]
        return None

    def fuzzy_match_catalogue(
        self, query: str, threshold: float = 0.5, top_k: int = 5
    ) -> list[RankedCandidate]:
        """Fuzzy match a query string against all test names, aliases, and write-ins."""
        norm_query = self.normalize_term(query)
        if not norm_query:
            return []

        candidates: list[RankedCandidate] = []
        seen_ids: set[str] = set()

        # 1. Exact alias match
        canonical_id = self.resolve_alias(norm_query)
        if canonical_id and canonical_id in self.catalogue:
            item = self.catalogue[canonical_id]
            candidates.append(
                RankedCandidate(
                    value=item.label,
                    canonical_id=canonical_id,
                    score=1.0,
                    tier=1,
                    reason="exact_alias_match",
                )
            )
            seen_ids.add(canonical_id)

        # 2. Search catalogue labels and aliases
        for fid, item in self.catalogue.items():
            if fid in seen_ids:
                continue
            best_sim = token_similarity(norm_query, item.label)
            for alias in item.aliases:
                sim = token_similarity(norm_query, alias)
                if sim > best_sim:
                    best_sim = sim
            if best_sim >= threshold:
                candidates.append(
                    RankedCandidate(
                        value=item.label,
                        canonical_id=fid,
                        score=round(best_sim, 3),
                        tier=_tier_from_score(best_sim),
                        reason="fuzzy_catalogue_match",
                    )
                )

        # 3. Search common write-ins
        for write_in in self.write_ins:
            sim = token_similarity(norm_query, write_in)
            if sim >= threshold:
                candidates.append(
                    RankedCandidate(
                        value=write_in.title(),
                        canonical_id=None,
                        score=round(sim, 3),
                        tier=_tier_from_score(sim),
                        reason="fuzzy_write_in_match",
                    )
                )

        # Sort descending by score
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top_k]

    def assume(
        self,
        field_id: str,
        raw_text: str,
        context: dict[str, Any],
        top_k: int = 5,
    ) -> list[RankedCandidate]:
        """Bayesian ranking of HTR candidate hypotheses given already-read marks."""
        ticked_ids = context.get("ticked_ids", [])
        implied = self.implied_tests(ticked_ids)

        # If matching write-in field (OTHERS)
        if field_id == "others":
            candidates = self.fuzzy_match_catalogue(raw_text, threshold=0.35, top_k=top_k * 2)
            # Boost score if candidate is medically related to already ticked tests
            boosted: list[RankedCandidate] = []
            for c in candidates:
                score = c.score
                reason = c.reason
                if c.canonical_id and c.canonical_id in implied:
                    score = min(1.0, score + 0.15)
                    reason += " [boosted_by_profile]"
                boosted.append(
                    c.model_copy(
                        update={
                            "score": round(score, 3),
                            "tier": _tier_from_score(score),
                            "reason": reason,
                        }
                    )
                )
            boosted.sort(key=lambda x: x.score, reverse=True)
            return boosted[:top_k]

        # For tube count fields (e.g., tube_edta, tube_cb)
        if field_id in self.tube_field_to_tube:
            tube_name = self.tube_field_to_tube[field_id]
            expected_tubes = self.calculate_expected_tubes(ticked_ids)
            expected_count = expected_tubes.get(tube_name, 0)
            clean_digits = "".join(ch for ch in raw_text if ch.isdigit())
            obs_count = int(clean_digits) if clean_digits else None

            if obs_count == expected_count:
                score = 0.95
                tier = 1
                reason = f"matches_expected_{tube_name}_count"
            elif obs_count is not None and abs(obs_count - expected_count) <= 1:
                score = 0.70
                tier = 2
                reason = f"near_expected_{tube_name}_count"
            else:
                score = 0.40
                tier = 3
                reason = f"expected_{tube_name}_{expected_count}_got_{obs_count}"

            return [
                RankedCandidate(
                    value=str(obs_count if obs_count is not None else expected_count),
                    canonical_id=field_id,
                    score=score,
                    tier=tier,
                    reason=reason,
                )
            ]

        # Default fallback
        return self.fuzzy_match_catalogue(raw_text, top_k=top_k)

    def validate_request(
        self,
        ticked_ids: list[str],
        observed_tubes: dict[str, int | None] | None = None,
        write_ins: list[str] | None = None,
    ) -> ValidationResult:
        """Validate order consistency between ticked tests, profiles, and sample tubes."""
        observed_tubes = observed_tubes or {}
        write_ins = write_ins or []

        implied = self.implied_tests(ticked_ids)
        all_ordered = sorted(set(ticked_ids) | implied)
        expected_tubes = self.calculate_expected_tubes(ticked_ids)

        warnings: list[str] = []
        discrepancies: list[str] = []

        # 1. Redundant standalone tests when parent profile is ordered
        for fid in ticked_ids:
            if fid in implied and fid not in self.profile_bundles:
                label = self.catalogue[fid].label if fid in self.catalogue else fid
                warnings.append(
                    f"Test '{label}' is redundantly checked; already included in ordered profile."
                )

        # 2. Check tube count consistency against expected tubes
        for tube_type, exp_count in expected_tubes.items():
            obs = observed_tubes.get(tube_type)
            if obs is not None and obs < exp_count:
                discrepancies.append(
                    f"Tube shortage: Expected >= {exp_count} '{tube_type}' tube(s) for ordered tests, but observed {obs}."
                )

        # 3. Check for empty request
        if not ticked_ids and not write_ins:
            warnings.append("No tests or profiles were selected on this request sheet.")

        confidence = 1.0
        if discrepancies:
            confidence = max(0.40, confidence - 0.25 * len(discrepancies))
        if warnings:
            confidence = max(0.60, confidence - 0.05 * len(warnings))

        return ValidationResult(
            is_valid=len(discrepancies) == 0,
            ticked_tests=ticked_ids,
            implied_tests=sorted(implied),
            all_ordered_tests=all_ordered,
            expected_tubes=expected_tubes,
            observed_tubes=observed_tubes,
            warnings=warnings,
            discrepancies=discrepancies,
            confidence=round(confidence, 3),
        )
