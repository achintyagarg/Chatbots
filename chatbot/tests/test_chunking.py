"""
Chunker tests.

Retrieval quality is bounded by chunking, and a bad chunker fails silently --
it returns plausible-looking chunks that just retrieve badly. These tests
pin the properties that would otherwise degrade without anyone noticing.
"""

from __future__ import annotations

import pytest

from ingestion.chunking import (
    Chunk,
    chunk_document,
    estimate_tokens,
    make_chunk_id,
    split_sentences,
)
from ingestion.loaders import Document, Section


def sentences(n: int, word: str = "alpha", words_each: int = 10) -> str:
    """n sentences of predictable length, each ending in a period."""
    return " ".join(
        f"{word.capitalize()}{i} " + " ".join([word] * (words_each - 2)) + " end."
        for i in range(n)
    )


def doc(*sections: Section, source: str = "spec.md", title: str = "Spec") -> Document:
    return Document(source=source, title=title, sections=list(sections))


class TestTokenEstimate:
    def test_scales_with_length(self):
        assert estimate_tokens("hello world") < estimate_tokens(sentences(5))

    def test_empty_is_zero(self):
        assert estimate_tokens("") == 0

    def test_long_words_cost_more_than_short(self):
        assert estimate_tokens("antidisestablishmentarianism") > estimate_tokens("cat")


class TestSentenceSplitting:
    def test_splits_on_terminal_punctuation(self):
        assert len(split_sentences("One thing. Two things! Three things?")) == 3

    def test_does_not_split_on_decimal_or_abbreviation(self):
        # A false boundary cuts a sentence in half, which is worse than a missed one.
        assert len(split_sentences("Version 1.5 shipped today.")) == 1

    def test_ignores_empty_fragments(self):
        assert split_sentences("   ") == []


class TestBoundaryPreservation:
    def test_short_document_is_one_chunk(self):
        chunks = chunk_document(doc(Section(text="A short paragraph.")))
        assert len(chunks) == 1
        assert chunks[0].text == "A short paragraph."

    def test_chunks_end_on_sentence_boundaries(self):
        chunks = chunk_document(doc(Section(text=sentences(80))), target_tokens=100)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.text.rstrip().endswith("."), f"mid-sentence split: {chunk.text[-60:]!r}"

    def test_no_chunk_is_empty_or_whitespace(self):
        chunks = chunk_document(doc(Section(text=sentences(40))), target_tokens=60)
        for chunk in chunks:
            assert chunk.text.strip()

    def test_all_source_text_survives(self):
        """Every sentence must appear somewhere; chunking must not drop content."""
        text = sentences(30)
        chunks = chunk_document(doc(Section(text=text)), target_tokens=80)
        combined = " ".join(c.text for c in chunks)
        for sentence in split_sentences(text):
            assert sentence in combined, f"lost: {sentence!r}"


class TestTokenBudget:
    def test_chunks_respect_target(self):
        target = 100
        chunks = chunk_document(doc(Section(text=sentences(60))), target_tokens=target, overlap_tokens=0)
        for chunk in chunks:
            # One unit may push past the target; two-plus means packing is broken.
            assert chunk.token_count <= target * 2

    def test_oversized_sentence_is_hard_split(self):
        """A single 'sentence' with no punctuation still has to be broken up."""
        giant = " ".join(["word"] * 2000)
        chunks = chunk_document(doc(Section(text=giant)), target_tokens=50)
        assert len(chunks) > 1
        for chunk in chunks:
            # Hard split must land on word boundaries, never mid-word.
            assert all(w == "word" for w in chunk.text.split())


class TestOverlap:
    def test_consecutive_chunks_share_text(self):
        chunks = chunk_document(
            doc(Section(text=sentences(60))), target_tokens=100, overlap_tokens=30
        )
        assert len(chunks) > 1
        first_sentences = set(split_sentences(chunks[0].text))
        second_sentences = set(split_sentences(chunks[1].text))
        assert first_sentences & second_sentences, "no overlap between adjacent chunks"

    def test_zero_overlap_shares_nothing(self):
        chunks = chunk_document(
            doc(Section(text=sentences(60))), target_tokens=100, overlap_tokens=0
        )
        first = set(split_sentences(chunks[0].text))
        second = set(split_sentences(chunks[1].text))
        assert not (first & second)

    def test_overlap_never_duplicates_a_whole_chunk(self):
        """A too-large overlap must not make chunk N+1 a superset of chunk N."""
        chunks = chunk_document(
            doc(Section(text=sentences(40))), target_tokens=100, overlap_tokens=100
        )
        for a, b in zip(chunks, chunks[1:]):
            assert a.text != b.text


class TestHeadingGrouping:
    def test_different_headings_never_merge(self):
        chunks = chunk_document(
            doc(
                Section(text="Install with pip.", heading_path=("Setup",)),
                Section(text="Call the run method.", heading_path=("Usage",)),
            )
        )
        assert len(chunks) == 2
        assert {c.heading for c in chunks} == {"Setup", "Usage"}

    def test_same_heading_sections_pack_together(self):
        chunks = chunk_document(
            doc(
                Section(text="First para.", heading_path=("Setup",)),
                Section(text="Second para.", heading_path=("Setup",)),
            )
        )
        assert len(chunks) == 1
        assert "First para." in chunks[0].text and "Second para." in chunks[0].text

    def test_heading_breadcrumb_is_joined(self):
        chunks = chunk_document(
            doc(Section(text="Body.", heading_path=("Guide", "Setup", "Windows")))
        )
        assert chunks[0].heading == "Guide > Setup > Windows"


class TestEmbedText:
    def test_includes_title_and_heading(self):
        """A chunk retrieved in isolation must still say what it is about."""
        chunks = chunk_document(doc(Section(text="Body text.", heading_path=("Setup",))))
        embed_text = chunks[0].embed_text
        assert "Spec" in embed_text and "Setup" in embed_text and "Body text." in embed_text

    def test_display_text_excludes_breadcrumb(self):
        chunks = chunk_document(doc(Section(text="Body text.", heading_path=("Setup",))))
        assert chunks[0].text == "Body text."


class TestStableIds:
    def test_same_input_same_ids(self):
        source = doc(Section(text=sentences(40)))
        first = chunk_document(source, target_tokens=80)
        second = chunk_document(source, target_tokens=80)
        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]

    def test_ids_unique_within_document(self):
        chunks = chunk_document(doc(Section(text=sentences(60))), target_tokens=60)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_different_sources_get_different_ids(self):
        a = chunk_document(doc(Section(text="Same text."), source="a.md"))
        b = chunk_document(doc(Section(text="Same text."), source="b.md"))
        assert a[0].chunk_id != b[0].chunk_id

    def test_id_is_deterministic_across_processes(self):
        # sha256, not hash() -- Python's hash is salted per process.
        assert make_chunk_id("spec.md", 0) == make_chunk_id("spec.md", 0)
        assert make_chunk_id("spec.md", 0) != make_chunk_id("spec.md", 1)


class TestEdgeCases:
    def test_empty_document(self):
        assert chunk_document(doc()) == []

    def test_whitespace_only_section(self):
        assert chunk_document(doc(Section(text="   \n\n  "))) == []

    def test_single_page_pdf_style_document(self):
        chunks = chunk_document(
            doc(Section(text="Only content.", heading_path=("Abstract",), page=1))
        )
        assert len(chunks) == 1
        assert chunks[0].page == 1

    def test_page_number_is_carried_through(self):
        chunks = chunk_document(
            doc(
                Section(text="Page three text.", heading_path=("Results",), page=3),
            )
        )
        assert chunks[0].page == 3

    def test_metadata_uses_sentinel_for_missing_page(self):
        """Chroma metadata rejects None, so page must degrade to -1."""
        chunks = chunk_document(doc(Section(text="No page.")))
        assert chunks[0].to_metadata()["page"] == -1

    def test_metadata_values_are_all_scalars(self):
        chunks = chunk_document(doc(Section(text="Body.", heading_path=("A", "B"), page=2)))
        for value in chunks[0].to_metadata().values():
            assert isinstance(value, (str, int, float, bool))


class TestLargeDocument:
    def test_many_sections_across_headings(self):
        sections = [
            Section(text=sentences(10), heading_path=(f"Section {i}",), page=i)
            for i in range(20)
        ]
        chunks = chunk_document(doc(*sections), target_tokens=120)
        assert len(chunks) >= 20
        assert len({c.chunk_id for c in chunks}) == len(chunks)
        assert all(c.text.strip() for c in chunks)
