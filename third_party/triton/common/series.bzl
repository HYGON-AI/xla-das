# Copyright 2026 The OpenXLA Authors. All Rights Reserved.
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
# =============================================================================

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
    "//third_party/triton:common/hcu-adaptation-part1.patch",
    "//third_party/triton:common/hcu-adaptation-part2.patch",
    # Add new patches just above this line
]
