# Copyright 2026 Ram Charan Satya Sai Teja Polisetti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Regression tests for cvs-health/langfair issue #247.

The gendered possessives ``his``/``her`` were substituted with a fixed 1:1
token swap (``his`` -> ``hers``, ``her`` -> ``him``). That is only correct
when the possessive is used as an independent/object pronoun. When used as
a possessive *determiner* before a noun ("his car", "her car") the swap
produced ungrammatical counterfactuals ("hers car", "him car"), confounding
the fairness signal of downstream assessments.

The fix (``context_sensitive_pronouns`` in ``_sub_from_dict``) disambiguates
with an NLTK POS-tag lookahead:
  - ``his`` + noun phrase -> ``her`` ; otherwise -> ``hers``
  - ``her`` + noun phrase -> ``his`` ; otherwise -> ``him``
"""

import pytest

from langfair.constants.word_lists import (
    FEMALE_WORDS,
    GENDER_NEUTRAL_WORDS,
    MALE_WORDS,
)
from langfair.generator import CounterfactualGenerator


@pytest.fixture(scope="module")
def cfgen():
    return CounterfactualGenerator(langchain_llm=None)


def _prompts(cfgen, text):
    return cfgen.create_prompts(prompts=[text], attribute="gender")


class TestIssue247Repro:
    """Exact reproduction cases from cvs-health/langfair#247."""

    def test_his_determiner_maps_to_her(self, cfgen):
        result = _prompts(cfgen, "His car was parked outside.")
        assert result["female_prompt"] == ["her car was parked outside."]

    def test_his_pronoun_still_maps_to_hers(self, cfgen):
        result = _prompts(cfgen, "That car is his.")
        assert result["female_prompt"] == ["that car is hers."]

    def test_her_determiner_maps_to_his(self, cfgen):
        result = _prompts(cfgen, "Her car was parked outside.")
        assert result["male_prompt"] == ["his car was parked outside."]

    def test_her_object_pronoun_still_maps_to_him(self, cfgen):
        result = _prompts(cfgen, "She was proud of her.")
        assert result["male_prompt"] == ["he was proud of him."]

    def test_issue_example_full(self, cfgen):
        result = _prompts(
            cfgen,
            "His wife was proud of him.",
        )
        # "wife" is already female-coded, so the female counterfactual keeps it;
        # the male counterfactual flips it to "husband" (spouse word-list fix).
        assert result["female_prompt"] == ["her wife was proud of her."]
        assert result["male_prompt"] == ["his husband was proud of him."]


class TestDeterminerDisambiguation:
    """Modifiers between the possessive and its head noun."""

    def test_adjective_between(self, cfgen):
        result = _prompts(cfgen, "His old car broke down.")
        assert result["female_prompt"] == ["her old car broke down."]

    def test_cardinal_between(self, cfgen):
        result = _prompts(cfgen, "Her 3 cats were sleeping.")
        assert result["male_prompt"] == ["his 3 cats were sleeping."]

    def test_participle_between(self, cfgen):
        result = _prompts(cfgen, "His running shoes were new.")
        assert result["female_prompt"] == ["her running shoes were new."]

    def test_ditransitive_object_pronoun_not_a_determiner(self, cfgen):
        # "her" is the indirect object here; "flowers" is not its noun phrase.
        result = _prompts(cfgen, "I gave her flowers.")
        assert result["male_prompt"] == ["i gave him flowers."]

    def test_object_complement_not_a_determiner(self, cfgen):
        # "her" is the object; "president" is an object complement, not
        # a noun phrase headed by "her".
        result = _prompts(cfgen, "They elected her president.")
        assert result["male_prompt"] == ["they elected him president."]

    def test_prepositional_object_pronoun(self, cfgen):
        result = _prompts(cfgen, "This gift is for her.")
        assert result["male_prompt"] == ["this gift is for him."]

    def test_predicate_pronoun_with_trailing_clause(self, cfgen):
        result = _prompts(cfgen, "The decision was his to make.")
        assert result["female_prompt"] == ["the decision was hers to make."]

    def test_sentence_initial_determiner(self, cfgen):
        result = _prompts(cfgen, "Her mother called yesterday.")
        assert result["male_prompt"] == ["his father called yesterday."]

    def test_identity_outputs_unchanged(self, cfgen):
        # Same-gender outputs must be identical to the (lowercased) input.
        result = _prompts(cfgen, "His car was parked outside.")
        assert result["male_prompt"] == ["his car was parked outside."]
        result = _prompts(cfgen, "Her car was parked outside.")
        assert result["female_prompt"] == ["her car was parked outside."]


class TestSpouseWordLists:
    """Spouse-relation nouns added to the gender word lists (issue #247 fix #1)."""

    def test_wife_husband_counterfactual(self, cfgen):
        result = _prompts(cfgen, "Her husband was proud of her.")
        assert result["female_prompt"] == ["her wife was proud of her."]

    def test_husband_wife_counterfactual(self, cfgen):
        result = _prompts(cfgen, "His wife was proud of him.")
        assert result["male_prompt"] == ["his husband was proud of him."]

    def test_plural_spouses(self, cfgen):
        result = _prompts(cfgen, "Their wives arrived early.")
        # "their"/"wives": wives -> husbands in the male counterfactual
        assert result["male_prompt"] == ["their husbands arrived early."]

    def test_neutral_spouse(self, cfgen):
        neutral = cfgen.neutralize_tokens(
            texts=["His wife was proud of him."], attribute="gender"
        )
        assert neutral == ["their spouse was proud of them."]

    def test_girlfriend_typo_fixed(self):
        assert "girlfriend" in FEMALE_WORDS
        assert "girfriend" not in FEMALE_WORDS

    def test_word_lists_stay_positionally_aligned(self):
        assert len(MALE_WORDS) == len(FEMALE_WORDS) == len(GENDER_NEUTRAL_WORDS)
        assert (
            MALE_WORDS.index("husband")
            == FEMALE_WORDS.index("wife")
            == GENDER_NEUTRAL_WORDS.index("spouse")
        )
        assert (
            MALE_WORDS.index("husbands")
            == FEMALE_WORDS.index("wives")
            == GENDER_NEUTRAL_WORDS.index("spouses")
        )


class TestFallbackBehavior:
    """When POS tagging is unavailable, the old flat mapping must hold."""

    def test_fallback_to_flat_mapping(self, cfgen, monkeypatch):
        monkeypatch.setattr(cfgen, "_ensure_pos_tagger", lambda: False)
        result = _prompts(cfgen, "His car was parked outside.")
        assert result["female_prompt"] == ["hers car was parked outside."]

    def test_custom_dict_keeps_flat_mapping(self, cfgen):
        result = cfgen.create_prompts(
            prompts=["His car was parked outside."],
            custom_dict={"male": ["his"], "female": ["hers"]},
        )
        assert result["female_prompt"] == ["hers car was parked outside."]
