#!/usr/bin/env bash
# download_youtube_training_data.sh
#
# Download BWF official match videos from YouTube using yt-dlp.
# Saves 720p MP4 files to backend/ml/data/raw/youtube/
#
# Usage:
#   chmod +x backend/ml/download_youtube_training_data.sh
#   ./backend/ml/download_youtube_training_data.sh
#
# Prerequisites:
#   pip install yt-dlp
#   brew install ffmpeg   (or apt-get install ffmpeg)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/data/raw/youtube"
mkdir -p "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Curated BWF official channel playlists
# These are publicly available BWF World Tour / Super Series playlists on
# the BWF TV YouTube channel (https://www.youtube.com/@BWFTV).
# ---------------------------------------------------------------------------
PLAYLISTS=(
    # BWF World Tour Finals highlights
    "https://www.youtube.com/playlist?list=PLEx5khR4g7PJnN4SOBT3cLm2Z3ALRQVYP"
    # BWF Super 1000 - All England highlights
    "https://www.youtube.com/playlist?list=PLEx5khR4g7PLNx3OczwT4K2jyAFBRxwgR"
    # BWF World Championships match videos
    "https://www.youtube.com/playlist?list=PLEx5khR4g7PK3Xk8e4C7kLtM-Xk9bF2Ig"
    # Thomas & Uber Cup
    "https://www.youtube.com/playlist?list=PLEx5khR4g7PJwf6zXKSQSPzLVaHmxJFZn"
)

echo "=== BWF Match Video Downloader ==="
echo "Output directory: ${OUTPUT_DIR}"
echo ""

for PLAYLIST_URL in "${PLAYLISTS[@]}"; do
    echo "Downloading playlist: ${PLAYLIST_URL}"
    yt-dlp \
        --format "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]" \
        --merge-output-format mp4 \
        --output "${OUTPUT_DIR}/%(playlist_title)s/%(playlist_index)03d_%(title)s.%(ext)s" \
        --restrict-filenames \
        --no-overwrites \
        --continue \
        --ignore-errors \
        --sleep-interval 2 \
        --max-sleep-interval 5 \
        --concurrent-fragments 4 \
        --write-info-json \
        --write-thumbnail \
        "${PLAYLIST_URL}" || {
            echo "  WARNING: playlist download had errors — continuing with next playlist"
        }
    echo ""
done

echo "=== Download complete ==="
echo "Files saved to: ${OUTPUT_DIR}"
echo ""
echo "Next step: run backend/ml/download_dataset.py with --skip-download"
echo "to convert these videos to TrackNetV4 training format."
echo "(You will need to add Label.csv annotation files first.)"
