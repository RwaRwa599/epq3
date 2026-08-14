"""Unit and integration tests for Block 2 Clinical Knowledge Graph."""

from __future__ import annotations

import pytest
from med_doc.kg import KnowledgeGraph
from med_doc.paths import DEFAULT_KG, DEFAULT_TEMPLATE, V0_KG, V1_TEMPLATE
from med_doc.template import load_template


@pytest.fixture
def kg_v1() -> KnowledgeGraph:
    return KnowledgeGraph.load(DEFAULT_KG)


@pytest.fixture
def kg_v0() -> KnowledgeGraph:
    return KnowledgeGraph.load(V0_KG)


def test_kg_loads_and_covers_all_template_fields(kg_v1: KnowledgeGraph, kg_v0: KnowledgeGraph):
    """Every checkbox field in the canonical templates must exist in the Knowledge Graph."""
    v1_template = load_template(V1_TEMPLATE)
    v0_template = load_template(DEFAULT_TEMPLATE)

    # v1 verification (138 checkboxes)
    v1_checkboxes = [f.field_id for f in v1_template.checkbox_fields()]
    assert len(v1_checkboxes) == 138
    for fid in v1_checkboxes:
        item = kg_v1.get_item(fid)
        assert item is not None, f"Field '{fid}' from v1 template missing in v1 Knowledge Graph"
        assert item.field_id == fid
        assert len(item.label) > 0

    # v0 verification (124 checkboxes)
    v0_checkboxes = [f.field_id for f in v0_template.checkbox_fields()]
    assert len(v0_checkboxes) == 124
    for fid in v0_checkboxes:
        item = kg_v0.get_item(fid)
        assert item is not None, f"Field '{fid}' from v0 template missing in v0 Knowledge Graph"


def test_profile_bundle_expansion(kg_v1: KnowledgeGraph):
    """Profiles must expand to their clinical component tests."""
    lipid_components = kg_v1.expand_profile("profile_lipid")
    assert "chol_total" in lipid_components
    assert "hdl" in lipid_components
    assert "ldl" in lipid_components
    assert "triglycerides" in lipid_components

    liver_components = kg_v1.expand_profile("profile_liver")
    assert "alt" in liver_components
    assert "ast" in liver_components
    assert "alp" in liver_components
    assert "bilirubin_total" in liver_components

    diabetes_components = kg_v1.expand_profile("profile_diabetes")
    assert "hba1c" in diabetes_components
    assert "glucose_fasting" in diabetes_components

    # Test implied_tests
    implied = kg_v1.implied_tests(["profile_lipid", "profile_diabetes"])
    assert "chol_total" in implied
    assert "hba1c" in implied


def test_calculate_expected_tubes(kg_v1: KnowledgeGraph):
    """Tube requirements must match physical laboratory protocols."""
    # Hematology test -> EDTA
    tubes_cbc = kg_v1.calculate_expected_tubes(["cbc"])
    assert tubes_cbc == {"EDTA": 1}

    # Chemistry tests -> Clotted Blood (CB)
    tubes_chem = kg_v1.calculate_expected_tubes(["alt", "creatinine", "chol_total"])
    assert tubes_chem == {"CB": 1}

    # Combined panel -> EDTA + CB + Fl
    tubes_combo = kg_v1.calculate_expected_tubes(["cbc", "alt", "glucose_fasting"])
    assert tubes_combo == {"EDTA": 1, "CB": 1, "Fl": 1}

    # Urine test -> Urine container
    tubes_urine = kg_v1.calculate_expected_tubes(["urinalysis"])
    assert tubes_urine == {"Urine": 1}

    # Pap smear & UBT
    tubes_pap_ubt = kg_v1.calculate_expected_tubes(["pap_smear", "urea_breath_test"])
    assert tubes_pap_ubt == {"pap": 1, "UBT": 1}


def test_alias_and_abbreviation_resolution(kg_v1: KnowledgeGraph):
    """Clinical aliases, acronyms, and abbreviations must resolve to canonical IDs."""
    assert kg_v1.resolve_alias("SGPT") == "alt"
    assert kg_v1.resolve_alias("ALT") == "alt"
    assert kg_v1.resolve_alias("SGOT") == "ast"
    assert kg_v1.resolve_alias("AST") == "ast"
    assert kg_v1.resolve_alias("HbA1c") == "hba1c"
    assert kg_v1.resolve_alias("VDRL") == "rpr"
    assert kg_v1.resolve_alias("RPR") == "rpr"
    assert kg_v1.resolve_alias("TSH") == "tsh"
    assert kg_v1.resolve_alias("FT4") == "ft4"


def test_fuzzy_match_catalogue(kg_v1: KnowledgeGraph):
    """Fuzzy matching must handle typos and partial write-in queries."""
    matches = kg_v1.fuzzy_match_catalogue("cultur and sensitivity", top_k=3)
    assert len(matches) > 0
    assert "culture" in matches[0].value.lower()
    assert matches[0].score >= 0.80

    matches_lipid = kg_v1.fuzzy_match_catalogue("cholestrol total", top_k=3)
    assert len(matches_lipid) > 0
    assert matches_lipid[0].canonical_id == "chol_total"


def test_assume_prior_ranking(kg_v1: KnowledgeGraph):
    """assume() must rank candidates and boost medically related tests."""
    # Context with profile_lipid checked
    context = {"ticked_ids": ["profile_lipid"]}
    ranked = kg_v1.assume("others", "triglycerides", context, top_k=3)
    assert len(ranked) > 0
    top = ranked[0]
    assert top.canonical_id == "triglycerides"
    assert top.tier == 1
    assert "boosted" in top.reason or top.score >= 0.90

    # Tube count assume
    context_tubes = {"ticked_ids": ["cbc", "alt"]}
    tube_ranked = kg_v1.assume("tube_edta", "1", context_tubes)
    assert len(tube_ranked) == 1
    assert tube_ranked[0].value == "1"
    assert tube_ranked[0].tier == 1


def test_validation_engine_discrepancy_detection(kg_v1: KnowledgeGraph):
    """Validation engine must detect tube shortages and redundant test selections."""
    # Case 1: Valid order with matching tubes
    res_valid = kg_v1.validate_request(
        ticked_ids=["cbc", "alt"],
        observed_tubes={"EDTA": 1, "CB": 1},
    )
    assert res_valid.is_valid is True
    assert len(res_valid.discrepancies) == 0
    assert res_valid.confidence >= 0.95

    # Case 2: Tube shortage (CB expected for alt, but 0 observed)
    res_shortage = kg_v1.validate_request(
        ticked_ids=["cbc", "alt"],
        observed_tubes={"EDTA": 1, "CB": 0},
    )
    assert res_shortage.is_valid is False
    assert any("CB" in d for d in res_shortage.discrepancies)
    assert res_shortage.confidence < 0.90

    # Case 3: Redundant component test checked alongside parent profile
    res_redundant = kg_v1.validate_request(
        ticked_ids=["profile_lipid", "chol_total"],
        observed_tubes={"CB": 1},
    )
    assert res_redundant.is_valid is True
    assert any("redundant" in w for w in res_redundant.warnings)

    # Case 4: Completely blank sheet
    res_blank = kg_v1.validate_request(ticked_ids=[])
    assert any("No tests" in w for w in res_blank.warnings)
