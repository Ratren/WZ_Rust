#!/usr/bin/env bash
# build z wektorami 512-bit (AVX-512). Użycie: scripts/avx512.sh bench|verify [opcje]
set -euo pipefail

if ! grep -qw avx512f /proc/cpuinfo; then
    echo "błąd: ten procesor nie obsługuje AVX-512" >&2
    exit 1
fi

root="$(cd "$(dirname "$0")/.." && pwd)"
# osobny katalog na każdą maszynę (target-cpu=native)
target="$root/target-avx512-$(hostname -s)"
RUSTFLAGS="-C target-cpu=native -C target-feature=-prefer-256-bit" \
    cargo build --release --quiet --manifest-path "$root/Cargo.toml" --target-dir "$target"
exec "$target/release/wz" "$@"
