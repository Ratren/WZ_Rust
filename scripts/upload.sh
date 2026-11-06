#!/usr/bin/env bash
# Wysyła kod projektu na serwer do ~/WZ_Rust (bez katalogów budowania, .git i wyników).
# Użycie: scripts/upload.sh [host]   (domyślnie: miranda)
set -euo pipefail

host="${1:-miranda}"
root="$(cd "$(dirname "$0")/.." && pwd)"

rsync -av --exclude '/target*' --exclude /.git --exclude /wyniki \
    "$root/" "$host:WZ_Rust/"
