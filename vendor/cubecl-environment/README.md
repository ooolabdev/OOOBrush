# CubeCL environment

This crate contains the build script and library from the pinned CubeCL revision `46752244`. The standalone manifest expands the upstream workspace settings and uses rusqlite 0.41.1 to bundle a patched SQLite. [Upstream source](https://github.com/tracel-ai/cubecl/tree/467522442c7b68f02c7ff0c84b2e249e36d8da91/crates/cubecl-environment) · [RustSec advisory](https://rustsec.org/advisories/RUSTSEC-2026-0459)

Remove the workspace patch when updating to a compatible CubeCL revision with patched SQLite.
