# BLS-SkipDFS Trajectory Analyzer

`bls_analyze` is a standalone C++ tool that reproduces the BLS + Skip-DFS
algorithm for fast cluster detection on molecular dynamics trajectories. It
consumes common GROMACS formats (XTC/TRR/TNG) alongside GRO/PDB snapshots,
voxelises the selected atoms into a lattice-aware occupancy grid, and reports
per-frame connectivity metrics that can be compared directly against PLUMED
outputs.

## Key capabilities

- **Flexible I/O** – modular trajectory readers with optional `xdrfile` (XTC/TRR)
  and `tng_io` (TNG) back-ends; text readers for GRO/PDB frames are always
  available.
- **PLUMED-style configuration** – a single `BLS ...` block defines groups,
  voxel spacing, lattice/centering, skip length, and the metrics to emit.
- **BLS seed pre-filter** – enumerates lattice points for cubic, hexagonal, and
  triclinic families with P/F/I centerings, scales them via the requested dNN,
  then triggers Skip-DFS refinement only on occupied seeds.
- **Skip-DFS refinement** – non-recursive DFS with 6/18/26 connectivity, PBC aware
  ray-casting, and support for ANY/ALL occupancy rules with optional cutoff
  padding.
- **Rich instrumentation** – CSV rows and JSON lines record frame ID, box
  resolution, seed statistics, cluster counts, and timing; optional comparison
  against PLUMED COLVAR/CSV files yields accuracy and speed-up summaries.
- **Parallel voxelisation** – OpenMP (when enabled) accelerates rasterisation and
  seed checks without affecting determinism.

## Building

```bash
cmake -S . -B build
cmake --build build
```

Optional dependencies can be toggled at configure time:

- `-DUSE_XDRFILE=ON` – link against `libxdrfile` to enable XTC/TRR support.
- `-DUSE_TNG=ON` – link against `libtng_io` to enable TNG support.

The resulting binaries are placed under `build/`:

- `bls_analyze` – command-line analyzer.
- `bls_tests` – unit test harness covering lattice scaling, connectivity modes,
  PBC handling, and deterministic runs.

Run the tests with `ctest --test-dir build`.

A minimal Docker recipe (`Dockerfile`) is provided to produce a clean Ubuntu
container with the analyzer pre-built.

## Configuration

The analyzer consumes a PLUMED-style control file. A minimal example is bundled
at `cpp/examples/bls.in`:

```plumed
BLS ...
  GROUP ATOMS=all
  BOX AUTO
  GRID_SPACING 0.25
  LATTICE cubic
  CENTERING F
  CONNECTIVITY 6
  DNN 0
  ALPHA 0.7
  RADII 1.5,2.0
  SKIP 3
  OUTPUT NCLUSTERS,MAX_CLUSTER,SEED_HITS,SEEDS,REFINED_VOXELS
... BLS
```

Key directives:

- `GROUP` – select atoms (`ATOMS=all`, `index:1-100`, `name:OW`, or combinations
  separated by `|`).
- `BOX` – `AUTO` uses trajectory boxes; otherwise provide explicit `XLO/XHI` etc.
- `GRID_SPACING` – voxel size in length units.
- `LATTICE` + `CENTERING` – choose unit basis and centering offsets.
- `DNN`/`RADII`/`ALPHA` – control the nearest-neighbour spacing used to scale the
  lattice (automatic when `DNN=0`).
- `SKIP` – maximum Skip-DFS ray length.
- `CONNECTIVITY` – neighbourhood (6/18/26).
- `OCCUPANCY` + `CUTOFF` – ANY/ALL occupancy rule with optional radius padding.
- `OUTPUT` – metrics to report (all core metrics are always recorded; this list
  controls optional extras such as cluster sizes in JSON).

## Command-line usage

```bash
./build/bls_analyze \
  --traj traj.xtc \
  --conf cpp/examples/bls.in \
  --top system.pdb \
  --out metrics.csv \
  --json metrics.json \
  --stride 5 \
  --compare-plumed plumed_colvar.dat
```

Available flags:

| Option | Description |
| ------ | ----------- |
| `--traj` | Trajectory file (XTC/TRR/TNG/GRO/PDB). |
| `--conf` | PLUMED-style configuration block. |
| `--top` | Optional topology (GRO/PDB) for named selections. |
| `--out` | CSV metrics path. |
| `--json` | JSON-lines metrics path. |
| `--stride` | Process every *N*-th frame (also multiplied by the config STRIDE). |
| `--start/--stop` | Frame index range (inclusive). Use `--stop inf` for no upper bound. |
| `--threads` | OpenMP threads (ignored if OpenMP disabled at build time). |
| `--format` | Override automatic trajectory format detection. |
| `--compare-plumed` | Reference PLUMED CSV/COLVAR for accuracy and timing comparisons. |

The analyzer prints per-frame summaries, writes CSV/JSON outputs when requested,
reports aggregate runtime/memory, and (if provided) prints comparison statistics
versus PLUMED including mean absolute and RMS errors and Kendall τ for maximum
cluster ranking.

## Tests

Unit tests live in `cpp/tests/TestMain.cpp` and validate:

- Correct FCC nearest-neighbour scaling.
- Connectivity differences between 6- and 26-neighbour schemes.
- Periodic boundary handling across wrapped voxels.
- Deterministic BLS/Skip-DFS execution across repeated runs.

Execute them via `ctest` or directly with `./build/bls_tests`.

## Project layout

```
cpp/
├── src/
│   ├── bls/            # BLS orchestrator and metrics
│   ├── config/         # PLUMED-style parser
│   ├── grid/           # Voxel grid + rasteriser
│   ├── io/             # Trajectory readers and factory
│   ├── lattice/        # Basis and seed enumerator helpers
│   ├── refine/         # Skip-DFS implementation
│   └── util/           # Logging, math utilities, timers
├── tests/              # Standalone test executable
└── examples/           # Sample configuration
```

All code follows a PLUMED-friendly structure so the analyzer can be embedded or
cross-validated in future PLUMED modules without significant refactoring.

