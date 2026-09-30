from verl.experimental.plugin_grpo.hint_quality import score_hint_fields


QUESTION = (
    "In '28 Best Academic Search Engines That make your research easier 2025', "
    "what is the reported content volume of Science.gov?"
)


def test_valid_three_field_hint_gets_full_reward():
    result = score_hint_fields(
        ("Science.gov", "reported content volume", "2025 academic search engines"),
        QUESTION,
        "Earlier searches did not locate the Science.gov entry.",
    )

    assert result["valid"] is True
    assert result["reward"] == 1.0
    assert result["violations"] == []


def test_real_meta_language_failure_gets_negative_reward():
    result = score_hint_fields(
        (
            "Extracting content volume reported figure",
            "So I'm trying here",
            "I see there's no content",
        ),
        QUESTION,
        "failed history",
    )

    assert result["valid"] is False
    assert result["reward"] < 0.0
    assert "no_self_talk" in result["violations"]
    assert "entity_anchor" in result["violations"]


def test_unanchored_entity_and_repeated_fields_are_penalized():
    unanchored = score_hint_fields(
        ("unrelated company", "chief executive", "2011 appointment"),
        QUESTION,
        "failed history",
    )
    repeated = score_hint_fields(
        ("Science.gov", "content volume", "content volume"),
        QUESTION,
        "failed history",
    )

    assert unanchored["reward"] < 0.0
    assert "entity_anchor" in unanchored["violations"]
    assert repeated["reward"] < 0.0
    assert "disambiguator" in repeated["violations"]
    assert "distinct" in repeated["violations"]
