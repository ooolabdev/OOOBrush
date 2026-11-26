# Maintaining OOOBrush against upstream Brush

Keep `ooolabdev/OOOBrush` as the product fork and `ArthurBrussee/brush` as its upstream. Use small, topic-specific Git commits as the maintained patch series rather than keeping another copy of the implementation in a directory of `.patch` files. A Git commit can be reviewed, exported or reapplied when needed. Do not rewrite the published history of `main` just to reorganize older patches.

## Downstream boundaries

| Area | Main implementation | Upstream-facing integration |
| --- | --- | --- |
| Geometry switches, weights and defaults | `crates/brush-process/src/geometry_options.rs` | One flattened geometry field in `ProcessConfig`; flat CLI, `args.txt` and JSON names stay stable. |
| Geometry input preparation | `crates/brush-process/src/geometry_inputs.rs` | A call during scene preparation in `train_stream.rs`; pass the resolved weights to the initial and LOD trainers. |
| Depth, normals and free-space sidecars | `crates/brush-dataset/src/geometry_sidecar.rs` | The existing `LoadImage` geometry fields and methods; RGB and mask loading remain in `load_image.rs`. |
| Geometry objective | `crates/brush-train/src/geometry.rs` | `GeometryStep` selects the render pass, splits its output and adds the existing losses in `SplatTrainer::step`. |
| Shortest-axis normal rendering and gradients | `crates/brush-render/` | The existing additional render channels and backward kernels. Review these together when upstream changes rasterization or the render contract. |
| CLI diagnostics and releases | `apps/brush-cli/`, process logging and `.github/workflows/cli-diagnostics.yml` | Preserve stdout/stderr, completion/error behavior and the `ooo-v*` release workflow. |

The three geometry losses are orientation (0.05), free-space opacity (0.02) and ray depth dispersion (0.01), with the existing 500-step warm-up and 500-step ramp. Calibrated depth regression and relative depth correlation remain absent. The refactor changes the internal Rust configuration shape, not the public CLI flags, flat saved keys, sidecar formats or PLY format. Keep post-training multi-view pruning and the editing threshold in OOOSplat, separate from the training objective.

The older geometry implementation is recorded in `d92e5ad`. The generic argument-merge fix (`c76a3af`), independent geometry controls (`9d79664`) and subsequent module extraction are separate commits, so future conflict resolution does not mix CLI parsing with the geometry objective. Future changes should follow the same separation. The historical renderer changes in `d92e5ad` still need review when upstream changes renderer internals; the module extraction does not remove that dependency.

## Integrating an official update

For a fresh clone, configure the official remote once. If `upstream` already exists, check its URL instead of adding it again. Enable Git's local reuse of recorded conflict resolutions (`rerere`).

```powershell
git remote -v
git remote add upstream https://github.com/ArthurBrussee/brush.git
git config --local rerere.enabled true
```

The November 26, 2026 integration reference is important for the first merge: the shared historical ancestor is `ce76c88`. Official changes #554 (`64b8533`) and #556 (`1388f74`) were already adapted onto the fork as `6378a76` and `d51454d`. They may appear as differences on both sides even though their functionality was imported. The fetched reference at the time of this refactor was `d43bd1e` (#566); that reference has **not** been merged or tested by this refactor.

Start with a clean worktree containing the intended downstream `main`. Fetch and inspect before choosing the official target commit. Review the actual changes against the previously adapted commits, rather than blindly cherry-picking the entire log again.

```powershell
git status --short --branch
git fetch --no-tags upstream main
git merge-base main upstream/main
git log --left-right --cherry-pick --oneline main...upstream/main
$upstreamCommit = git rev-parse upstream/main
$shortCommit = git rev-parse --short $upstreamCommit
$integrationBranch = "integration/upstream-$shortCommit"
git switch -c $integrationBranch main
git merge --no-ff $upstreamCommit
```

Review conflicts on the integration branch. Prefer the official implementation for unchanged functionality and reconnect the downstream modules at their integration points. In particular, check the render channel layout, camera/geometry map contract and loss gradients together. Preserve the existing fork CLI contract, generic `args.txt` merge rules, diagnostic messages, pinned dependency fixes, and platform/release workflows. Do not use an `ours` merge to declare unintegrated official changes as incorporated. Record the exact official SHA and any conflict resolutions in the integration commit or PR.

Only after the merge is resolved and the validation below passes, fast-forward `main` to the reviewed integration branch. If `main` has moved, integrate it and repeat validation rather than forcing a push.

```powershell
git switch main
git merge --ff-only $integrationBranch
```

Commit and push intentionally. A normal `main` push runs the existing three-platform build workflow; it does not create a Release. Publishing remains a separate release-tag or manual workflow action.

## Regression checks before updating main or releasing

Use the toolchain and locked dependencies from the selected integration commit. For the current fork:

```powershell
cargo fmt --all --check
cargo test --locked -p brush-cli -p brush-process -p brush-train -p brush-dataset --lib -- --test-threads=1
cargo build --locked -p brush-cli --bin brush-cli
```

Some training/dataset tests exercise the GPU. Run them on a suitable local device, and use the existing `cli-diagnostics.yml` workflow for hosted, GPU-independent diagnostics and Windows/Linux/macOS builds. Check the actual CLI help, progress and final export behavior on the target engine binary.

For a repeatable local GPU check, use the fixture generator/runner below. It needs Python with NumPy and Pillow (already used by `scripts/geometry`), and a native `brush-cli` build. It does not download a model, prepare a real dataset, or modify existing files. Always choose a new output directory.

```powershell
python -B scripts/geometry/smoke_options.py --brush .\target\debug\brush-cli.exe --output .\target\geometry-smoke-current
```

On Linux/macOS, use `--brush ./target/debug/brush-cli` instead. An explicit `--backend vulkan` is optional for local comparisons; otherwise the script preserves the engine's normal backend selection and any existing `WGPU_BACKEND` setting. It sets `RUST_LOG=info` and `RUST_BACKTRACE=1` only for its child training processes.

Seven 510-step runs check RGB-only training, each individual geometry loss with only its needed inputs, prepared defaults, custom weights, and override of `args.txt` without losing its unrelated SH setting. The last successful case also checks the original `brush_process::train_stream` log target under a narrow `RUST_LOG` filter. The runs cross the 500-step warm-up and check the final progress, export success, completion status, and complete finite PLY payload. A separate missing-map run must fail with a nonzero exit and no successful completion or final PLY. Each case leaves its input, original log and output in the chosen directory; `summary.json` summarizes passing cases from the completed suite. Keep a previous binary and run it into another fresh directory when comparing changes.

If upstream changes the renderer, also run `cargo test --locked -p brush-render --lib -- --test-threads=1` on a GPU, including the shortest-axis normal and gradient tests. The small smoke fixture is a compatibility check, not a visual-quality benchmark. Before releasing rendering or training changes, compare the same fixed real scenes, split, seed and settings for PSNR and foreground/free-space violations; inspect the resulting PLYs for floaters and orientation artifacts. Compare numerical metrics rather than requiring byte-identical GPU output. Do not silently change weights or thresholds while checking an upstream update.
