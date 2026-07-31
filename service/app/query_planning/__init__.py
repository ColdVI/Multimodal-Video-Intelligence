"""Additive query-planning contracts (parser, ontology, relaxation) for POST /search.

Nothing in this package is imported by default when parser_mode=none/filter_relaxation_
mode=off (the existing defaults) beyond these lightweight dataclasses -- heavy providers
(transformers_local, vllm_openai_compatible) are imported lazily inside rules.py/llm.py
only when actually selected. See docs/architecture/QUERY_PLANNING.md.
"""
