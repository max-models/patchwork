"""The tag of patchwork's own profiling regions, which are off unless enabled.

Every region is named ``"patchwork.<operation>"`` and tagged with
:data:`PROFILE_TAGS`. The package registers the tag as opt-in with
scope-profiler, so an application that profiles itself records these regions
only when it asks for them (``enable_tags=["patchwork"]``; see the README).
"""

#: The scope-profiler tags of every patchwork region.
PROFILE_TAGS = ("patchwork",)
