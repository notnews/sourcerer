"""Package-level invariants: the model pin and the lazy API imports."""

import subprocess
import sys

import sourcerer


def test_importing_the_package_does_not_pull_in_the_sdk() -> None:
    """``import sourcerer`` must not import ``anthropic``.

    Eager re-exports previously made ``import sourcerer.schema`` fail with
    ``ModuleNotFoundError: anthropic`` wherever the SDK was absent, so the test
    suite could not even be collected there. Run in a subprocess: this process
    has almost certainly imported the SDK already via another test.
    """
    code = (
        "import sys; import sourcerer; "
        "assert 'anthropic' not in sys.modules, "
        "'importing sourcerer pulled in the anthropic SDK'"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_schema_is_usable_without_touching_the_api_modules() -> None:
    claim = sourcerer.Claim(
        text="Poland has welcomed nearly two million Ukrainians.",
        claimer="John Berman",
        claimer_role=sourcerer.schema.ClaimerRole.ANCHOR,
        claim_type=sourcerer.schema.ClaimType.FACTUAL,
    )
    assert claim.to_dict()["claimer_role"] == "anchor"


def test_lazy_names_resolve_and_unknown_ones_raise() -> None:
    assert callable(sourcerer.extract_claims)
    try:
        sourcerer.no_such_export
    except AttributeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("unknown attribute should raise AttributeError")


def test_every_advertised_export_resolves() -> None:
    """``__all__`` is hand-written, so it can drift from what exists."""
    for name in sourcerer.__all__:
        assert getattr(sourcerer, name) is not None, name


def test_model_pin_is_a_current_model() -> None:
    """A retired pin is invisible until a request is made and 404s.

    The previous pin (``claude-sonnet-4-20250514``) sat in fifteen function
    signatures and went stale silently. Assert the shape we expect rather than
    the exact string, so a deliberate upgrade does not fail the suite but an
    accidental revert to a dated snapshot does.
    """
    assert sourcerer.MODEL.startswith("claude-")
    assert not sourcerer.MODEL[-8:].isdigit(), f"{sourcerer.MODEL} looks like a dated snapshot; use an undated alias"
