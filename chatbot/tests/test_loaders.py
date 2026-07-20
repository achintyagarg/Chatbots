"""Loader tests: the structure recovered here is what chunking relies on."""

from __future__ import annotations

import pytest

from ingestion.loaders import (
    _looks_like_heading,
    load_directory,
    load_markdown,
    load_path,
    load_text,
)


def write(tmp_path, name: str, content: str):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


class TestMarkdown:
    def test_heading_stack_nests(self, tmp_path):
        path = write(
            tmp_path,
            "guide.md",
            "# Guide\n\nIntro text.\n\n## Setup\n\nInstall it.\n\n### Windows\n\nUse the installer.\n",
        )
        document = load_markdown(path, "guide.md")
        paths = [s.heading_path for s in document.sections]
        assert ("Guide",) in paths
        assert ("Guide", "Setup") in paths
        assert ("Guide", "Setup", "Windows") in paths

    def test_dedent_pops_back_to_sibling(self, tmp_path):
        """A second ## must replace the first, not nest under it."""
        path = write(
            tmp_path,
            "g.md",
            "# T\n\n## A\n\nAlpha.\n\n### A1\n\nDeep.\n\n## B\n\nBeta.\n",
        )
        document = load_markdown(path, "g.md")
        beta = next(s for s in document.sections if "Beta." in s.text)
        assert beta.heading_path == ("T", "B")

    def test_title_comes_from_h1(self, tmp_path):
        path = write(tmp_path, "g.md", "# Real Title\n\nBody.\n")
        assert load_markdown(path, "g.md").title == "Real Title"

    def test_title_falls_back_to_filename(self, tmp_path):
        path = write(tmp_path, "no-heading.md", "Just body text.\n")
        assert load_markdown(path, "no-heading.md").title == "no-heading"

    def test_paragraphs_become_separate_sections(self, tmp_path):
        path = write(tmp_path, "g.md", "# T\n\nOne.\n\nTwo.\n\nThree.\n")
        document = load_markdown(path, "g.md")
        assert len(document.sections) == 3


class TestText:
    def test_splits_paragraphs(self, tmp_path):
        path = write(tmp_path, "n.txt", "First para.\n\nSecond para.\n")
        document = load_text(path, "n.txt")
        assert [s.text for s in document.sections] == ["First para.", "Second para."]

    def test_no_headings(self, tmp_path):
        path = write(tmp_path, "n.txt", "Body.")
        assert load_text(path, "n.txt").sections[0].heading_path == ()

    def test_empty_file_yields_no_sections(self, tmp_path):
        path = write(tmp_path, "n.txt", "   \n\n  \n")
        assert load_text(path, "n.txt").sections == []


class TestHeadingHeuristic:
    @pytest.mark.parametrize(
        "line", ["Introduction", "3.1 Results", "Appendix A", "Related Work"]
    )
    def test_accepts_headings(self, line):
        assert _looks_like_heading(line)

    @pytest.mark.parametrize(
        "line",
        [
            "This is an ordinary sentence that ends with a period.",
            "lowercase start",
            "",
            "Introduction:",  # trailing colon reads as a label, not a heading
            "A very long line with far too many words to plausibly be a heading here",
        ],
    )
    def test_rejects_non_headings(self, line):
        assert not _looks_like_heading(line)

    def test_wrapped_body_line_is_not_a_heading(self):
        """
        The classic false positive: a wrapped line looks exactly like a
        heading until you notice the next line continues its sentence.
        """
        assert not _looks_like_heading(
            "Tier 2 support responds within 90 minutes during business",
            "hours and within 6 hours otherwise.",
        )

    def test_real_heading_survives_lookahead(self):
        assert _looks_like_heading("Contact Escalation", "Tier 2 support responds.")

    def test_heading_followed_by_blank_line(self):
        assert _looks_like_heading("Error Codes", "")


class TestDispatch:
    def test_unsupported_extension_raises(self, tmp_path):
        path = write(tmp_path, "data.xlsx", "x")
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_path(path)

    def test_source_is_relative_to_root(self, tmp_path):
        nested = tmp_path / "docs" / "sub"
        nested.mkdir(parents=True)
        path = write(nested, "a.md", "# A\n\nBody.\n")
        document = load_path(path, root=tmp_path)
        assert document.source.replace("\\", "/") == "docs/sub/a.md"

    def test_directory_walk_skips_unsupported(self, tmp_path):
        write(tmp_path, "a.md", "# A\n\nBody.\n")
        write(tmp_path, "b.txt", "Body.\n")
        write(tmp_path, "c.xlsx", "ignored")
        documents = load_directory(tmp_path)
        assert {d.source for d in documents} == {"a.md", "b.txt"}
