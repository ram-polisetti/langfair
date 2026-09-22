# Changelog — ram-polisetti/langfair (fork of cvs-health/langfair)

All notable changes to this fork are documented here. Upstream releases are
tracked at https://github.com/cvs-health/langfair/releases.

## [Unreleased]

### Fixed
- **Context-sensitive gender possessive substitution in `CounterfactualGenerator`**
  (upstream issue [cvs-health/langfair#247](https://github.com/cvs-health/langfair/issues/247)).
  The gendered possessives `his`/`her` were substituted with a fixed 1:1 token
  swap (`his` -> `hers`, `her` -> `him`), which is only grammatical when the
  possessive is used as an independent/object pronoun ("that car is his").
  When used as a possessive *determiner* before a noun — the overwhelmingly
  common case ("his car", "her mother") — the swap produced ungrammatical
  counterfactuals ("hers car", "him car"). Because the broken variant
  systematically attracts grammar-correction behavior from the model under
  test, this confounded the fairness signal of counterfactual assessments.
  - `_sub_from_dict` accepts `context_sensitive_pronouns=True` (enabled for
    the built-in `attribute="gender"` path in `create_prompts`; custom user
    dictionaries keep the flat 1:1 mapping).
  - An NLTK POS-tag lookahead disambiguates determiner use (scan forward past
    adjectives/adverbs/participles/cardinals for the head noun) from pronoun
    use. Determiner uses now map `his` -> `her` and `her` -> `his`; pronoun
    uses keep the flat mapping (`his` -> `hers`, `her` -> `him`).
  - A preceding verb marks the ditransitive object-pronoun case for `her`
    ("gave her flowers" -> "him", not "his"), so previously-correct outputs
    do not regress.
  - If the POS tagger data is unavailable (e.g. no network to download it),
    the code falls back to the previous flat mapping instead of failing.
  - Known limitation: `her` after a verb but heading a noun phrase
    ("saw her patient") is classified as an object pronoun, as before.
- **Spouse-relation nouns added to the gender word lists** (the "quick win"
  from issue #247): `wife`/`wives` <-> `husband`/`husbands`, with neutral
  equivalents `spouse`/`spouses`. All three positional lists stay aligned.
- **Typo fix**: `girfriend` -> `girlfriend` in `FEMALE_WORDS` (matches
  upstream PR #248).

### Added
- `tests/test_counterfactual_pronouns.py`: 22 regression tests covering the
  issue #247 repro cases (both directions), determiner disambiguation with
  intervening modifiers, ditransitive/object-complement guards, spouse-noun
  substitution, the typo fix, word-list alignment, the no-tagger fallback,
  and the custom-dict flat-mapping path.

### Test report (this fork, 2026-09-22)
- New tests: 22/22 pass.
- Runnable existing tests (`test_display.py`, `test_adversarial.py`): 33 pass;
  2 failures in `test_adversarial.py` are pre-existing on the pristine
  upstream tree (verified via `git stash`) and unrelated to this change.
- 10 test modules require heavy optional dependencies (torch, detoxify,
  transformers, langchain-openai, …) not installed in this environment and
  could not be collected here; they are unaffected by this change
  (no modified code paths outside `CounterfactualGenerator.create_prompts`
  / `_sub_from_dict` and the gender word lists).
