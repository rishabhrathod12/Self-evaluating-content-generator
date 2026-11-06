import pytest
from app.evaluation.rubric import RUBRIC, get_item, rubric_text
from app.evaluation.schemas import CRITERIA


def test_rubric_matches_schema_criteria_in_order():
    assert tuple(item.name for item in RUBRIC) == CRITERIA


def test_every_item_has_concrete_conditions():
    for item in RUBRIC:
        assert len(item.pass_when) >= 3


def test_rubric_text_full_and_subset():
    full = rubric_text()
    for name in CRITERIA:
        assert name in full
    subset = rubric_text([CRITERIA[3]])
    assert CRITERIA[3] in subset
    assert CRITERIA[0] not in subset


def test_get_item_unknown_raises():
    with pytest.raises(KeyError):
        get_item("Vibes")
