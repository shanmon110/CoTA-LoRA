#!/usr/bin/env bash
# Download the CREMA-D audio (7,442 wav files, about 600 MB) from the official repository and verify every file.
#
#   bash experiments/hearside_pilot/get_cremad.sh            # writes /root/autodl-tmp/crema-d/AudioWAV
#
# The repository stores the audio in Git LFS. This script clones only the small pointer files, which record
# the SHA-256 and size of each clip, downloads each clip over HTTPS and checks it against its pointer.
# Running it again downloads only what is missing or damaged. No git-lfs installation is needed.
set -uo pipefail

DEST="${DEST:-/root/autodl-tmp/crema-d}"
REPO="${REPO:-https://github.com/CheyneyComputerScience/CREMA-D.git}"
COMMIT="${COMMIT:-1658cd342dff90010aa843eaeebd53610a08b1dc}"
MEDIA="${MEDIA:-https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/$COMMIT/AudioWAV}"
JOBS="${JOBS:-8}"
LIMIT="${LIMIT:-0}"                                   # for testing: only the first LIMIT clips
EXPECTED=7442

mkdir -p "$DEST/AudioWAV" || exit 1
if [ ! -d "$DEST/pointers/.git" ]; then
    rm -rf "$DEST/pointers"
    GIT_LFS_SKIP_SMUDGE=1 git clone -q --filter=blob:none --sparse --no-checkout "$REPO" "$DEST/pointers" \
        || { echo "FAILED: could not clone $REPO" >&2; exit 1; }
fi
( cd "$DEST/pointers" && git sparse-checkout set AudioWAV >/dev/null 2>&1 && GIT_LFS_SKIP_SMUDGE=1 git -c advice.detachedHead=false checkout -q "$COMMIT" ) \
    || { echo "FAILED: could not check out commit $COMMIT" >&2; exit 1; }

list="$DEST/expected.sha256"
for f in "$DEST"/pointers/AudioWAV/*.wav; do
    oid=$(sed -n 's/^oid sha256://p' "$f")
    [ -n "$oid" ] && echo "$oid  $(basename "$f")"
done > "$list"
[ "$LIMIT" -gt 0 ] && { head -n "$LIMIT" "$list" > "$list.part" && mv "$list.part" "$list"; EXPECTED=$LIMIT; }
count=$(wc -l < "$list")
[ "$count" -eq "$EXPECTED" ] || { echo "FAILED: the repository lists $count clips, expected $EXPECTED" >&2; exit 1; }

fetch() {   # sha256, file name
    local out="$DEST/AudioWAV/$2"
    [ -f "$out" ] && [ "$(sha256sum "$out" | cut -d' ' -f1)" = "$1" ] && return 0
    for attempt in 1 2 3 4; do
        curl -sfL --retry 3 --connect-timeout 20 --max-time 120 -o "$out.part" "$MEDIA/$2" \
            && [ "$(sha256sum "$out.part" | cut -d' ' -f1)" = "$1" ] && { mv "$out.part" "$out"; return 0; }
        sleep "$attempt"
    done
    rm -f "$out.part"
    echo "could not fetch $2" >&2
    return 1
}
export -f fetch
export DEST MEDIA
echo "downloading $count clips into $DEST/AudioWAV with $JOBS parallel jobs"
tr -s ' ' < "$list" | xargs -P "$JOBS" -n 2 bash -c 'fetch "$0" "$1"' || true

bad=$(cd "$DEST/AudioWAV" && sha256sum --quiet -c "$list" 2>/dev/null | wc -l)
have=$(find "$DEST/AudioWAV" -maxdepth 1 -name '*.wav' | wc -l)
if [ "$bad" -ne 0 ] || [ "$have" -ne "$EXPECTED" ]; then
    echo "INCOMPLETE: $have of $EXPECTED wav files present, $bad missing or damaged. Run this script again to continue." >&2
    exit 1
fi
echo "CREMA-D READY: $have wav files in $DEST/AudioWAV, every SHA-256 matches commit ${COMMIT:0:7}"
