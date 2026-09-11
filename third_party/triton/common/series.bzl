"""
Provides a list of patches that are applied internally and in oss.

IMPORTANT: GitHub contributions should not be adding patches to this list, as
the Google team does not have the bandwidth to handle their continuous updates
or upstreaming them. Please directly contribute your changes to
github.com/triton-lang/triton instead.

If you are fixing something in a BUILD file, please update the patch file in
third_party/triton/oss_only or add a patch there instead.

Modified by Hygon Information Technology Co., Ltd., 2026.
"""

common_patch_list = [
    "//third_party/triton:common/hcu-adaptation.patch",
    # Add new patches just above this line
]
