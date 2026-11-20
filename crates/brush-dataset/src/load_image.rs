use brush_render::AlphaMode;
use brush_vfs::BrushVfs;
use image::{DynamicImage, GrayImage, ImageBuffer};
use std::{
    io::{self, Cursor},
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::io::AsyncReadExt;

#[derive(Clone, Debug)]
pub struct LoadImage {
    vfs: Arc<BrushVfs>,
    path: PathBuf,
    mask_path: Option<PathBuf>,
    geometry_paths: [Option<PathBuf>; 3],
    pub depth_calibration: Option<crate::scene::DepthCalibration>,
    max_resolution: u32,
    alpha_mode: AlphaMode,
    invert_mask: bool,
    scale: f32,
}

impl PartialEq for LoadImage {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
            && self.mask_path == other.mask_path
            && self.invert_mask == other.invert_mask
            && self.max_resolution == other.max_resolution
            && self.scale == other.scale
            && self.geometry_paths == other.geometry_paths
            && self.depth_calibration == other.depth_calibration
    }
}

impl LoadImage {
    pub fn new(
        vfs: Arc<BrushVfs>,
        path: PathBuf,
        mask_path: Option<PathBuf>,
        max_resolution: u32,
        override_alpha_mode: Option<AlphaMode>,
        invert_mask: bool,
    ) -> Self {
        let alpha_mode = override_alpha_mode.unwrap_or_else(|| {
            if mask_path.is_some() {
                AlphaMode::Masked
            } else {
                AlphaMode::Transparent
            }
        });

        Self {
            vfs,
            path,
            mask_path,
            geometry_paths: Default::default(),
            depth_calibration: None,
            max_resolution,
            alpha_mode,
            invert_mask,
            scale: 1.0,
        }
    }

    pub async fn load(&self) -> image::ImageResult<DynamicImage> {
        let mut img_bytes = vec![];
        self.vfs
            .reader_at_path(&self.path)
            .await?
            .read_to_end(&mut img_bytes)
            .await?;
        let mut img = decode_with_cap(&img_bytes, &self.path, self.max_resolution)?;

        // Copy over mask.
        if let Some(mask_path) = &self.mask_path {
            // Add in alpha channel if needed to the image to copy the mask into.
            let mut masked_img = img.into_rgba8();
            let mut mask_bytes = vec![];
            self.vfs
                .reader_at_path(mask_path)
                .await?
                .read_to_end(&mut mask_bytes)
                .await?;
            let mask_img = image::load_from_memory(&mask_bytes)?;

            // Only one channel of the mask ever reaches the alpha channel, so
            // reduce it to 8bpp before doing any work on it. A grayscale mask
            // (the usual case) then costs no conversion at all, and the resize
            // below runs over one channel instead of three or four.
            let mut mask = if mask_img.color().has_alpha() {
                let rgba = mask_img.into_rgba8();
                let (w, h) = rgba.dimensions();
                let alpha = rgba.pixels().map(|p| p[3]).collect();
                GrayImage::from_raw(w, h, alpha).expect("one byte per pixel")
            } else {
                mask_img.into_luma8()
            };

            // Resize mask image if needed. This is allowed to squash the mask.
            if mask.dimensions() != masked_img.dimensions() {
                mask = image::imageops::resize(
                    &mask,
                    masked_img.width(),
                    masked_img.height(),
                    image::imageops::FilterType::Triangle,
                );
            }

            for (pixel, mask_pixel) in masked_img.pixels_mut().zip(mask.pixels()) {
                pixel[3] = if self.invert_mask {
                    255 - mask_pixel[0]
                } else {
                    mask_pixel[0]
                };
            }

            img = masked_img.into();
        }

        let scale = self.output_scale(img.width(), img.height());
        if scale < 1.0 {
            let new_w = (img.width() as f32 * scale).max(1.0) as u32;
            let new_h = (img.height() as f32 * scale).max(1.0) as u32;
            Ok(img.resize_exact(new_w, new_h, image::imageops::FilterType::Lanczos3))
        } else {
            Ok(img)
        }
    }

    /// Factor `load()` applies to a source of size `w`x`h`: the long edge is
    /// capped to `max_resolution` and multiplied by `scale`.
    fn output_scale(&self, w: u32, h: u32) -> f32 {
        let max = self.max_resolution;
        let cap = max as f32 / w.max(h).max(max) as f32;
        (cap * self.scale).min(1.0)
    }

    /// Read just the image dimensions from the file header, without decoding
    /// the pixels. Much cheaper than `load()` when only the size is needed
    /// (e.g. formats that carry intrinsics but not image dimensions).
    ///
    /// Reads the file in chunks and stops as soon as the header parses, so for
    /// typical formats only the first chunk is fetched rather than the whole
    /// (potentially many-MB) file. A truncated prefix only ever fails to parse
    /// (the dimension fields are reported once fully present), so a partial
    /// buffer can't yield wrong dimensions.
    pub async fn dimensions(&self) -> image::ImageResult<(u32, u32)> {
        let mut reader = self.vfs.reader_at_path(&self.path).await?;
        let dims = brush_vfs::read_until_parsed(&mut reader, 64 * 1024, |bytes| {
            image::ImageReader::new(Cursor::new(bytes))
                .with_guessed_format()
                .ok()
                .and_then(|r| r.into_dimensions().ok())
        })
        .await?;
        dims.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("could not determine image dimensions for {:?}", self.path),
            )
            .into()
        })
    }

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

    pub fn alpha_mode(&self) -> AlphaMode {
        self.alpha_mode
    }

    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    pub fn with_max_resolution(mut self, max_resolution: u32) -> Self {
        self.max_resolution = max_resolution;
        self
    }

    pub fn img_name(&self) -> String {
        Path::new(&self.path)
            .file_name()
            .expect("No file name for eval view.")
            .to_string_lossy()
            .to_string()
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Decode `bytes`, hinting `jpeg-decoder`'s IDCT scaler to land at or just
/// above `max_resolution` on the long edge for JPEG inputs — that cuts decode
/// cost by ~4-16× on oversized source images. Falls back to `image` for
/// non-JPEG files and for JPEG pixel formats we don't unpack directly.
fn decode_with_cap(
    bytes: &[u8],
    path: &Path,
    max_resolution: u32,
) -> image::ImageResult<DynamicImage> {
    let is_jpeg = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("jpg") || e.eq_ignore_ascii_case("jpeg"));
    if is_jpeg && let Some(img) = decode_jpeg_scaled(bytes, max_resolution) {
        return Ok(img);
    }
    image::load_from_memory(bytes)
}

fn decode_jpeg_scaled(bytes: &[u8], max_resolution: u32) -> Option<DynamicImage> {
    let mut decoder = jpeg_decoder::Decoder::new(Cursor::new(bytes));
    let target = max_resolution.min(u16::MAX as u32) as u16;
    decoder.scale(target, target).ok()?;
    let pixels = decoder.decode().ok()?;
    let info = decoder.info()?;
    let w = info.width as u32;
    let h = info.height as u32;
    match info.pixel_format {
        jpeg_decoder::PixelFormat::RGB24 => {
            ImageBuffer::from_raw(w, h, pixels).map(DynamicImage::ImageRgb8)
        }
        jpeg_decoder::PixelFormat::L8 => {
            ImageBuffer::from_raw(w, h, pixels).map(DynamicImage::ImageLuma8)
        }
        // CMYK32 / L16 are rare in photogrammetry data; fall back to image crate.
        _ => None,
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::LoadImage;
    use brush_vfs::BrushVfs;
    use image::{DynamicImage, GrayImage, Rgb, RgbImage, RgbaImage};
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

    /// Write an image + mask pair to a temp dir and load them back.
    async fn load_with_mask(mask: DynamicImage, invert: bool) -> RgbaImage {
        let dir = tempfile::tempdir().expect("temp dir");
        RgbImage::from_pixel(4, 2, Rgb([10, 20, 30]))
            .save(dir.path().join("img.png"))
            .expect("save image");
        mask.save(dir.path().join("mask.png")).expect("save mask");

        let vfs = Arc::new(BrushVfs::from_path(dir.path()).await.expect("vfs"));
        LoadImage::new(
            vfs,
            "img.png".into(),
            Some("mask.png".into()),
            1920,
            None,
            invert,
        )
        .load()
        .await
        .expect("load image")
        .into_rgba8()
    }

    #[tokio::test]
    async fn mask_becomes_alpha() {
        let mask = GrayImage::from_raw(4, 2, (0..8).map(|i| i * 30).collect()).expect("mask");
        let img = load_with_mask(mask.into(), false).await;
        let alpha: Vec<u8> = img.pixels().map(|p| p[3]).collect();
        assert_eq!(alpha, (0..8).map(|i| i * 30).collect::<Vec<_>>());
    }

    #[tokio::test]
    async fn inverted_mask_flips_alpha() {
        let mask = GrayImage::from_raw(4, 2, (0..8).map(|i| i * 30).collect()).expect("mask");
        let img = load_with_mask(mask.into(), true).await;
        let alpha: Vec<u8> = img.pixels().map(|p| p[3]).collect();
        assert_eq!(alpha, (0..8).map(|i| 255 - i * 30).collect::<Vec<_>>());
    }
}
