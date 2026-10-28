"""Patchwork's profiling regions: off by default, on with ``enable_tags``."""

from __future__ import annotations

import pytest
import scope_profiler as sp

from patchwork import AMRHierarchy


@pytest.fixture(autouse=True)
def _fresh_profiler():
    """No profiling state from other tests, and none left for them."""
    sp.reset()
    yield
    sp.reset()


def regions_of_a_halo_exchange(**setup_kwargs) -> set[str]:
    """The names of the regions recorded around one halo exchange."""
    sp.ProfileManager.setup(output=None, **setup_kwargs)
    mesh = AMRHierarchy(base_shape=(8, 8))
    mesh.add_patch(level=1, box=((4, 12), (4, 12)))
    mesh.allocate_field("u", halo=1)
    with sp.region("user.step"):
        mesh.exchange_halos("u")
    results = sp.finalize(verbose=False, return_results=True)
    return set(results.region_names)


def test_patchwork_regions_are_off_by_default():
    names = regions_of_a_halo_exchange()

    assert "user.step" in names
    assert not {name for name in names if name.startswith("patchwork.")}


def test_enable_tags_records_the_patchwork_regions():
    names = regions_of_a_halo_exchange(enable_tags=["patchwork"])

    assert {
        "user.step",
        "patchwork.exchange_halos",
        "patchwork.build_schedule",
    } <= names


def test_the_opt_in_tag_has_an_entry_point():
    """``scope-profiler tags`` finds the tag through the package's entry point."""
    from importlib.metadata import entry_points

    from patchwork._profiling import PROFILE_TAGS

    found = {
        entry_point.name: entry_point.value
        for entry_point in entry_points(group="scope_profiler.opt_in_tags")
    }
    assert found.get("patchwork") == "patchwork"
    assert "patchwork" in PROFILE_TAGS
