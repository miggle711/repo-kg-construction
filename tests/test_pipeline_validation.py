"""kg_construction#54's root-cause finding: TestContextValidator already
had checks (_check_seed_types, _check_seed_connectivity) that would have
caught the test-file-as-seed bug every time it happened, but nothing ever
surfaced it -- extract_and_validate only printed the report when
verbose=True, kg-test-generation's caller passed verbose=False and
discarded the returned report entirely, and is_valid was never checked
anywhere. The validator worked; nothing listened to it.

These tests cover _handle_validation_result, the helper that makes a
validation result impossible to silently ignore, without needing a real
KG build (which requires a network clone via RepoManager).
"""

import pytest

from kg_construction.pipeline import _handle_validation_result


class TestHandleValidationResult:
    def test_valid_result_does_not_raise_even_when_strict(self):
        _handle_validation_result(
            is_valid=True, report="all good", repo="psf/requests",
            commit="deadbeef", verbose=False, strict=True,
        )

    def test_invalid_result_raises_when_strict(self):
        with pytest.raises(ValueError, match="validation failed"):
            _handle_validation_result(
                is_valid=False, report="Disconnected seeds (1): foo",
                repo="psf/requests", commit="deadbeef",
                verbose=False, strict=True,
            )

    def test_invalid_result_does_not_raise_when_not_strict(self):
        # Default behavior stays non-blocking -- strict is opt-in so
        # existing interactive/exploratory callers aren't broken.
        _handle_validation_result(
            is_valid=False, report="Disconnected seeds (1): foo",
            repo="psf/requests", commit="deadbeef",
            verbose=False, strict=False,
        )

    def test_invalid_result_is_printed_even_when_verbose_is_false(self, capsys):
        # This is the actual kg_construction#54 bug: verbose=False (what
        # kg-test-generation's real pipeline call uses) used to discard
        # the report entirely, so a real validation error was never seen
        # by anyone -- not printed, not raised, not logged.
        _handle_validation_result(
            is_valid=False, report="Disconnected seeds (1): foo",
            repo="psf/requests", commit="deadbeef",
            verbose=False, strict=False,
        )
        captured = capsys.readouterr()
        assert "Disconnected seeds" in captured.out

    def test_valid_result_prints_nothing_extra_when_verbose_is_false(self, capsys):
        _handle_validation_result(
            is_valid=True, report="all good", repo="psf/requests",
            commit="deadbeef", verbose=False, strict=False,
        )
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_does_not_double_print_when_verbose_already_printed_it(self, capsys):
        # When verbose=True, extract_and_validate already printed `report`
        # itself before calling this helper -- it must not print again.
        _handle_validation_result(
            is_valid=False, report="Disconnected seeds (1): foo",
            repo="psf/requests", commit="deadbeef",
            verbose=True, strict=False,
        )
        captured = capsys.readouterr()
        assert captured.out == ""
