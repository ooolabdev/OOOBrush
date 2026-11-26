//! MoGe-2 sidecar path selection, decoding and training target preparation.
//!
//! This is a child module of `load_image` so its existing private file handles
//! remain private. RGB/mask loading and the public `LoadImage` API do not change.

use std::{io, path::PathBuf};

use image::DynamicImage;
use tokio::io::AsyncReadExt;

use super::LoadImage;

impl LoadImage {
    pub fn enable_geometry(
        &mut self,
        files: &std::collections::HashSet<PathBuf>,
        depth: bool,
        normals: bool,
        free_space: bool,
    ) -> io::Result<()> {
        let images = self
            .path
            .ancestors()
            .find(|p| p.file_name().is_some_and(|n| n == "images"))
            .ok_or_else(|| {
                io::Error::other("geometry supervision requires an images/ directory")
            })?;
        let relative = self
            .path
            .strip_prefix(images)
            .unwrap()
            .with_extension("png");
        for (slot, (enabled, folder)) in self.geometry_paths.iter_mut().zip([
            (depth, "depths"),
            (normals, "normals"),
            (free_space, "free-space"),
        ]) {
            if enabled {
                let path = images.with_file_name(folder).join(&relative);
                if !files.contains(&path) {
                    return Err(io::Error::new(
                        io::ErrorKind::NotFound,
                        format!("missing geometry map {}", path.display()),
                    ));
                }
                *slot = Some(path);
            }
        }
        Ok(())
    }

    pub async fn load_geometry(&self, w: u32, h: u32) -> io::Result<crate::scene::GeometryMaps> {
        use crate::scene::{GEOMETRY_STRIDE as STRIDE, GeometryMaps};
        use burn::tensor::TensorData;
        let mut maps = GeometryMaps::default();
        let gw = (w as usize).div_ceil(STRIDE);
        let gh = (h as usize).div_ceil(STRIDE);
        for (index, path) in self.geometry_paths.iter().enumerate() {
            let Some(path) = path else { continue };
            let mut bytes = Vec::new();
            self.vfs
                .reader_at_path(path)
                .await?
                .read_to_end(&mut bytes)
                .await?;
            let map = image::load_from_memory_with_format(&bytes, image::ImageFormat::Png)
                .map_err(|e| io::Error::other(format!("{}: {e}", path.display())))?;
            let (mw, mh) = (map.width(), map.height());
            if (mw as f64 / w as f64 * h as f64 - mh as f64).abs() > 2.0 {
                return Err(io::Error::other(format!(
                    "geometry map aspect mismatch: {}",
                    path.display()
                )));
            }
            let coords = (0..gh).flat_map(|y| {
                (0..gw).map(move |x| {
                    let map_pixel = |v: usize, src: u32, dst: u32| {
                        (((2 * v * STRIDE + 1) as u64 * dst as u64) / (2 * src as u64)) as u32
                    };
                    (map_pixel(x, w, mw), map_pixel(y, h, mh))
                })
            });
            match (index, map) {
                (0, DynamicImage::ImageLuma16(img)) => {
                    maps.depth = Some(TensorData::new(
                        coords
                            .map(|(x, y)| img.get_pixel(x, y)[0] as f32 / 65535.0)
                            .collect(),
                        [gh, gw],
                    ));
                }
                (2, DynamicImage::ImageLuma16(img)) => {
                    let calibration = self.depth_calibration.ok_or_else(|| {
                        io::Error::other("free-space targets require depth calibration")
                    })?;
                    let samples: Vec<u16> = coords.map(|(x, y)| img.get_pixel(x, y)[0]).collect();
                    let mut support: Vec<_> = samples
                        .iter()
                        .copied()
                        .filter(|&s| s > 0 && s < u16::MAX)
                        .collect();
                    if support.len() >= 16 && calibration.confidence > 0.0 {
                        let n = support.len();
                        let bins = [5, 35, 70].map(|p| *support.select_nth_unstable(n * p / 100).1);
                        let channels = samples
                            .into_iter()
                            .map(|s| {
                                if s == u16::MAX {
                                    0.0
                                } else {
                                    bins.partition_point(|&v| v <= s) as f32
                                }
                            })
                            .collect();
                        let planes = bins.map(|v| 0.9 * v as f32 / 65535.0 * calibration.scale);
                        maps.free_space = Some((planes, TensorData::new(channels, [gh, gw])));
                    }
                }
                (1, DynamicImage::ImageRgb8(img)) => {
                    let normals = coords
                        .flat_map(|(x, y)| {
                            let rgb = img.get_pixel(x, y).0;
                            if rgb == [0, 0, 0] {
                                [0.0; 3]
                            } else {
                                rgb.map(|v| v as f32 / 127.5 - 1.0)
                            }
                        })
                        .collect();
                    maps.normals = Some(TensorData::new(normals, [gh, gw, 3]));
                }
                _ => {
                    return Err(io::Error::other(format!(
                        "unexpected geometry PNG format: {}",
                        path.display()
                    )));
                }
            }
        }
        Ok(maps)
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::LoadImage;
    use brush_vfs::BrushVfs;
    use image::{Rgb, RgbImage};
    use std::sync::Arc;

    #[tokio::test]
    async fn loads_nested_geometry_and_preserves_invalid_pixels() {
        use image::{ImageBuffer, Luma};
        let dir = tempfile::tempdir().unwrap();
        for root in ["images", "depths", "normals"] {
            std::fs::create_dir_all(dir.path().join(root).join("cam1")).unwrap();
        }
        ImageBuffer::from_fn(8, 8, |x, y| {
            Luma([if x == 0 && y == 0 { 0u16 } else { 32768 }])
        })
        .save(dir.path().join("depths/cam1/view.png"))
        .unwrap();
        RgbImage::from_fn(8, 8, |x, y| {
            if x == 0 && y == 0 {
                Rgb([0, 0, 0])
            } else {
                Rgb([128, 128, 0])
            }
        })
        .save(dir.path().join("normals/cam1/view.png"))
        .unwrap();
        let vfs = Arc::new(BrushVfs::from_path(dir.path()).await.unwrap());
        let files = vfs.file_paths().collect();
        let mut input = LoadImage::new(vfs, "images/cam1/view.jpg".into(), None, 1920, None, false);
        let disabled = input.load_geometry(8, 8).await.unwrap();
        assert!(disabled.depth.is_none() && disabled.normals.is_none());
        assert!(
            input
                .enable_geometry(&Default::default(), true, false, false)
                .is_err()
        );
        input.enable_geometry(&files, true, true, false).unwrap();
        let maps = input.load_geometry(8, 8).await.unwrap();
        let z = maps.depth.unwrap();
        assert_eq!(z.shape.dims(), [2, 2]);
        let z = z.as_slice::<f32>().unwrap();
        assert_eq!(z[0], 0.0);
        assert!((z[1] - 0.5).abs() < 0.001);
        let n = maps.normals.unwrap();
        let n = n.as_slice::<f32>().unwrap();
        assert_eq!(&n[..3], &[0.0; 3]);
        assert_eq!(n[5], -1.0);
    }

    #[tokio::test]
    async fn free_space_planes_are_conservative_and_mask_invalid_depths() {
        use image::{ImageBuffer, Luma};
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("free-space")).unwrap();
        ImageBuffer::from_fn(8, 8, |x, y| {
            Luma([match x + 8 * y {
                0 => 0u16,
                63 => u16::MAX,
                i => i as u16 * 100,
            }])
        })
        .save(dir.path().join("free-space/view.png"))
        .unwrap();
        let vfs = Arc::new(BrushVfs::from_path(dir.path()).await.unwrap());
        let files = vfs.file_paths().collect();
        let mut input = LoadImage::new(vfs, "images/view.jpg".into(), None, 32, None, false);
        input.enable_geometry(&files, false, false, true).unwrap();
        input.depth_calibration = Some(crate::scene::DepthCalibration {
            scale: 2.,
            confidence: 1.,
        });
        let (planes, data) = input
            .load_geometry(32, 32)
            .await
            .unwrap()
            .free_space
            .unwrap();
        let bins = data.as_slice::<f32>().unwrap();
        assert_eq!((bins[0], bins[63]), (0., 0.));
        assert_eq!(bins[62], 3.);
        for (i, &bin) in bins.iter().enumerate().filter(|(_, b)| **b > 0.) {
            assert!(planes[bin as usize - 1] <= 0.9 * (i * 100) as f32 / 65535. * 2.);
        }
    }
}
