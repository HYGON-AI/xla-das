#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

if [[ -z "$ROCMINFO" || ! -x "$ROCMINFO" ]]; then
  for candidate in "${ROCM_PATH:-/opt/dtk}/bin/rocminfo" "/opt/dtk/bin/rocminfo"; do
    if [[ -x "$candidate" ]]; then
      ROCMINFO="$candidate"
      break
    fi
  done
fi

if [[ -z "$ROCMINFO" || ! -x "$ROCMINFO" ]]; then
  echo "ERROR: cannot locate rocminfo (looked in runfiles and \${ROCM_PATH:-/opt/dtk}/bin)." >&2
  echo "Refusing to run: without a device count, tests would run unlocked and share one HCU." >&2
  exit 1
fi

TF_GPU_COUNT=${TF_GPU_COUNT:-$("$ROCMINFO" | grep -c "Name: *gfx")}
TF_TESTS_PER_GPU=${TF_TESTS_PER_GPU:-8}
if [[ "$TF_GPU_COUNT" -eq 0 ]]; then
  echo "No HCU devices reported by ${ROCMINFO}." >&2
  exit 1
fi

# This function is used below in rlocation to check that a path is absolute
function is_absolute {
  [[ "$1" = /* ]] || [[ "$1" =~ ^[a-zA-Z]:[/\\].* ]]
}

export TF_PER_DEVICE_MEMORY_LIMIT_MB=${TF_PER_DEVICE_MEMORY_LIMIT_MB:-4096}

RUNFILES_MANIFEST_FILE="${TEST_SRCDIR}/MANIFEST"
function rlocation() {
  if is_absolute "$1" ; then
    # If the file path is already fully specified, simply return it.
    echo "$1"
  elif [[ -e "$TEST_SRCDIR/$1" ]]; then
    # If the file exists in the $TEST_SRCDIR then just use it.
    echo "$TEST_SRCDIR/$1"
  elif [[ -e "$RUNFILES_MANIFEST_FILE" ]]; then
    # If a runfiles manifest file exists then use it.
    echo "$(grep "^$1 " "$RUNFILES_MANIFEST_FILE" | sed 's/[^ ]* //')"
  fi
}

TEST_BINARY="$(rlocation $TEST_WORKSPACE/${1#./})"
shift

mkdir -p /var/lock
for j in `seq 0 $((TF_TESTS_PER_GPU-1))`; do
  for i in `seq 0 $((TF_GPU_COUNT-1))`; do
    exec {lock_fd}>/var/lock/gpulock${i}_${j} || exit 1
    if flock -n "$lock_fd";
    then
      (
        export CUDA_VISIBLE_DEVICES=$i
        export HIP_VISIBLE_DEVICES=$i
        echo "Running test $TEST_BINARY $* on HCU $HIP_VISIBLE_DEVICES"
        "$TEST_BINARY" $@
      )
      return_code=$?
      flock -u "$lock_fd"
      exec {lock_fd}>&-
      exit $return_code
    fi
    exec {lock_fd}>&-
  done
done

echo "Cannot find a free HCU to run the test $* on, exiting with failure..."
exit 1
