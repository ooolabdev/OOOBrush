//! Prepare only the geometry inputs required by the selected training losses.

use std::sync::Arc;

use anyhow::Context;
use brush_dataset::scene::Scene;
use brush_train::config::GeometryLossWeights;
use brush_vfs::BrushVfs;

use crate::geometry_options::GeometryOptions;

pub(crate) async fn prepare_for_training(
    vfs: &BrushVfs,
    scene: &mut Scene,
    options: &GeometryOptions,
    units_per_meter: f32,
) -> anyhow::Result<GeometryLossWeights> {
    let prepared = vfs
        .iter_files()
        .any(|p| p == std::path::Path::new("depth-calibration.json"));
    let geometry_weights = options.resolve(prepared)?;
    // Keep the existing target for consumers filtering the training stream by module.
    if geometry_weights.enabled() {
        prepare_geometry_inputs(vfs, scene, geometry_weights, units_per_meter).await?;
        log::info!(
            target: "brush_process::train_stream",
            "Geometry supervision: orientation={} free_space={} depth_distortion={} calibrated_depth=0 relative_depth=0 warmup=500 ramp=500",
            geometry_weights.orientation,
            geometry_weights.free_space,
            geometry_weights.depth_distortion
        );
    } else {
        log::info!(target: "brush_process::train_stream", "Geometry supervision off; using RGB training");
    }

    Ok(geometry_weights)
}

/// Load only the supervision inputs needed by the resolved losses. In
/// particular, depth dispersion is a render-only loss and must not activate
/// the `MoGe` file loader just to make the trainer's geometry branch reachable.
async fn prepare_geometry_inputs(
    vfs: &BrushVfs,
    scene: &mut Scene,
    weights: GeometryLossWeights,
    units_per_meter: f32,
) -> anyhow::Result<()> {
    if !weights.enabled() {
        return Ok(());
    }
    let orientation = weights.orientation > 0.0;
    let free_space = weights.free_space > 0.0;
    let calibration: Option<
        std::collections::HashMap<String, brush_dataset::scene::DepthCalibration>,
    > = if free_space {
        use tokio::io::AsyncReadExt;
        let mut bytes = Vec::new();
        vfs.reader_at_path(std::path::Path::new("depth-calibration.json"))
            .await
            .context("opening depth-calibration.json")?
            .read_to_end(&mut bytes)
            .await?;
        Some(serde_json::from_slice(&bytes).context("parsing depth-calibration.json")?)
    } else {
        None
    };
    let files = if orientation || free_space {
        vfs.file_paths().collect()
    } else {
        std::collections::HashSet::new()
    };
    for view in Arc::make_mut(&mut scene.views) {
        // Retain the camera-model scope of the existing geometry objective.
        anyhow::ensure!(
            view.camera.camera_model == brush_render::kernels::camera_model::CameraModel::Pinhole,
            "geometry supervision requires undistorted pinhole views"
        );
        if orientation || free_space {
            view.image
                .enable_geometry(&files, orientation, orientation, free_space)?;
        }
        if let Some(calibration) = &calibration {
            let key = view.image.path().to_string_lossy().replace('\\', "/");
            let mut value = *calibration
                .get(&key)
                .with_context(|| format!("missing depth calibration for {key}"))?;
            value.scale /= units_per_meter;
            anyhow::ensure!(
                value.scale.is_finite()
                    && value.scale > 0.0
                    && (0.0..=1.0).contains(&value.confidence),
                "invalid depth calibration for {key}"
            );
            view.image.depth_calibration = Some(value);
        }
    }
    Ok(())
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::*;
    use std::path::Path;

    async fn geometry_test_scene(root: &Path) -> anyhow::Result<(Arc<BrushVfs>, Scene)> {
        let vfs = Arc::new(BrushVfs::from_path(root).await?);
        let scene = Scene::new(vec![brush_dataset::scene::SceneView {
            camera: brush_render::camera::Camera::default(),
            image: brush_dataset::scene::LoadImage::new(
                vfs.clone(),
                "images/view.jpg".into(),
                None,
                32,
                None,
                false,
            ),
        }]);
        Ok((vfs, scene))
    }

    #[test]
    fn geometry_input_requirements_follow_only_enabled_losses() -> anyhow::Result<()> {
        let root =
            std::env::temp_dir().join(format!("brush-geometry-input-probe-{}", std::process::id()));
        std::fs::create_dir(&root)?;
        for name in ["images", "depths", "normals", "free-space"] {
            std::fs::create_dir(root.join(name))?;
        }
        std::fs::write(root.join("images/view.jpg"), b"fixture")?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let result = runtime.block_on(async {
            // Pin-hole RGB data without any geometry sidecars or manifest.
            let (vfs, mut scene) = geometry_test_scene(&root).await?;
            prepare_geometry_inputs(&vfs, &mut scene, GeometryLossWeights::default(), 1.0).await?;
            let distortion = GeometryLossWeights {
                depth_distortion: 0.005,
                ..Default::default()
            };
            prepare_geometry_inputs(&vfs, &mut scene, distortion, 1.0).await?;
            assert!(scene.views[0].image.depth_calibration.is_none());
            let orientation = GeometryLossWeights {
                orientation: 0.025,
                ..Default::default()
            };
            assert!(
                prepare_geometry_inputs(&vfs, &mut scene, orientation, 1.0)
                    .await
                    .is_err()
            );
            let free = GeometryLossWeights {
                free_space: 0.02,
                ..Default::default()
            };
            assert!(
                prepare_geometry_inputs(&vfs, &mut scene, free, 1.0)
                    .await
                    .is_err()
            );
            drop(scene);
            drop(vfs);

            // Orientation needs depth and normals, but not calibrated free space.
            std::fs::write(root.join("depths/view.png"), b"fixture")?;
            std::fs::write(root.join("normals/view.png"), b"fixture")?;
            let (vfs, mut scene) = geometry_test_scene(&root).await?;
            prepare_geometry_inputs(&vfs, &mut scene, orientation, 1.0).await?;
            assert!(scene.views[0].image.depth_calibration.is_none());
            drop(scene);
            drop(vfs);
            std::fs::remove_file(root.join("depths/view.png"))?;
            std::fs::remove_file(root.join("normals/view.png"))?;

            // Free-space training uses its prepared maps and scale without
            // forcing the raw depth/normal loader or the orientation pass.
            std::fs::write(root.join("free-space/view.png"), b"fixture")?;
            std::fs::write(
                root.join("depth-calibration.json"),
                br#"{"images/view.jpg":{"scale":4,"confidence":0.5}}"#,
            )?;
            let (vfs, mut scene) = geometry_test_scene(&root).await?;
            prepare_geometry_inputs(&vfs, &mut scene, free, 2.0).await?;
            let calibration = scene.views[0].image.depth_calibration.unwrap();
            assert_eq!(calibration.scale, 2.0);
            assert_eq!(calibration.confidence, 0.5);
            Ok::<(), anyhow::Error>(())
        });
        for file in [
            "images/view.jpg",
            "depths/view.png",
            "normals/view.png",
            "free-space/view.png",
            "depth-calibration.json",
        ] {
            let path = root.join(file);
            if path.exists() {
                std::fs::remove_file(path)?;
            }
        }
        for name in ["images", "depths", "normals", "free-space"] {
            std::fs::remove_dir(root.join(name))?;
        }
        std::fs::remove_dir(root)?;
        result
    }
}
