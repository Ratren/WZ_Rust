# wz

Rozkład WZ macierzy w Rust (sekwencyjnie i równolegle). Przepisane z wersji OpenMP z `Resources/WZ/testy.h`.

## Uruchamianie

```sh
cargo test
cargo wz verify
cargo wz bench
cargo wz bench --sizes 1000:4000:1000 --threads 1,2,4,8,16 --csv wyniki.csv
```

Domyślnie liczy w f64. Dla f32 albo obu naraz:

```sh
cargo wz bench --precision f32
cargo wz bench --precision both
cargo wz verify --precision f32
```

Reszta opcji: `cargo wz bench --help`.

Warianty: `seq`, `rayon`, `threads` (wątki z barierą), każdy z kernelem `simd` albo `scalar` (bez wektoryzacji).
Wyniki: mediana, średnia, odchylenie, min/max, GFLOPS i speedup względem seq.

## Serwer

- `scripts/upload.sh [host]` - wysyła kod przez rsync (domyślnie na miranda)
- na maszynie z NUMA najlepiej odpalać przez `numactl`, np. `numactl --physcpubind=0-19 --membind=0 ./target/release/wz bench ...`
- `scripts/avx512.sh bench ...` - to samo, ale z wektorami 512-bit (tylko na procesorach z AVX-512)

## Uwagi

- w oryginale w C pętla po k gubi ostatni krok dla nieparzystego n, tutaj jest poprawione
- generator losowy można podmienić: trait `RandomSource` w `src/generate.rs` + jedna linia w `input_matrix`
