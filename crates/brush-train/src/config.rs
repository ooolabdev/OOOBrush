use brush_render::gaussian_splats::SplatRenderMode;
use clap::Parser;
use serde::{Deserialize, Serialize};

/// Weights for the MoGe-2 geometry objective after warm-up.
pub const ORIENTATION_WEIGHT: f32 = 0.05;
pub const FREE_SPACE_WEIGHT: f32 = 0.02;
pub const DEPTH_DISTORTION_WEIGHT: f32 = 0.01;

/// Effective weights after the process-level switches and defaults have been
/// resolved. Zero disables a loss, including its additional rendering work.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GeometryLossWeights {
    pub orientation: f32,
    pub free_space: f32,
    pub depth_distortion: f32,
}

impl GeometryLossWeights {
    pub const fn standard() -> Self {
        Self {
            orientation: ORIENTATION_WEIGHT,
            free_space: FREE_SPACE_WEIGHT,
            depth_distortion: DEPTH_DISTORTION_WEIGHT,
        }
    }

    pub fn enabled(self) -> bool {
        self.orientation > 0.0 || self.free_space > 0.0 || self.depth_distortion > 0.0
    }

    pub(crate) fn for_step(
        self,
        step: u32,
        has_orientation_targets: bool,
        has_free_space_targets: bool,
    ) -> Self {
        let ramp = (step.saturating_sub(500) as f32 / 500.0).min(1.0);
        Self {
            orientation: if has_orientation_targets {
                self.orientation * ramp
            } else {
                0.0
            },
            free_space: if has_free_space_targets {
                self.free_space * ramp
            } else {
                0.0
            },
            depth_distortion: self.depth_distortion * ramp,
        }
    }
}

#[cfg(test)]
mod geometry_weight_tests {
    use super::*;

    #[test]
    fn geometry_weights_keep_the_existing_ramp_and_independent_targets() {
        let weights = GeometryLossWeights::standard();
        assert_eq!(
            weights.for_step(500, true, true),
            GeometryLossWeights::default()
        );
        assert_eq!(weights.for_step(1000, true, true), weights);
        assert_eq!(
            weights.for_step(750, true, true).orientation,
            ORIENTATION_WEIGHT * 0.5
        );
        assert_eq!(
            weights.for_step(750, true, true).free_space,
            FREE_SPACE_WEIGHT * 0.5
        );
        assert_eq!(
            weights.for_step(750, true, true).depth_distortion,
            DEPTH_DISTORTION_WEIGHT * 0.5
        );

        // Missing normals must not disable free space or depth dispersion.
        let no_normals = weights.for_step(1000, false, true);
        assert_eq!(no_normals.orientation, 0.0);
        assert_eq!(no_normals.free_space, FREE_SPACE_WEIGHT);
        assert_eq!(no_normals.depth_distortion, DEPTH_DISTORTION_WEIGHT);
        let no_maps = weights.for_step(1000, false, false);
        assert_eq!(no_maps.free_space, 0.0);
        assert_eq!(no_maps.depth_distortion, DEPTH_DISTORTION_WEIGHT);

        let orientation_only = GeometryLossWeights {
            orientation: 0.025,
            ..Default::default()
        };
        assert_eq!(
            orientation_only.for_step(1200, true, true),
            orientation_only
        );
        assert!(
            !GeometryLossWeights::default()
                .for_step(1200, true, true)
                .enabled()
        );
    }
}

#[derive(Clone, Parser, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct TrainConfig {
    /// Total number of steps to train for.
    #[arg(long, help_heading = "Training options", default_value = "30000")]
    pub total_train_iters: u32,

    #[arg(long, help_heading = "Training options")]
    pub render_mode: Option<SplatRenderMode>,

    /// Start learning rate for the mean parameters.
    #[arg(long, help_heading = "Training options", default_value = "2e-5")]
    pub lr_mean: f64,

    /// Start learning rate for the mean parameters.
    #[arg(long, help_heading = "Training options", default_value = "2e-7")]
    pub lr_mean_end: f64,

    /// How much noise to add to the mean parameters of low opacity gaussians.
    #[arg(long, help_heading = "Training options", default_value = "50.0")]
    pub mean_noise_weight: f32,

    /// Learning rate for the base SH (RGB) coefficients.
    #[arg(long, help_heading = "Training options", default_value = "2e-3")]
    pub lr_coeffs_dc: f64,

    /// How much to divide the learning rate by for higher SH orders.
    #[arg(long, help_heading = "Training options", default_value = "10.0")]
    pub lr_coeffs_sh_scale: f32,

    /// Learning rate for the opacity parameter.
    #[arg(long, help_heading = "Training options", default_value = "0.012")]
    pub lr_opac: f64,

    /// Learning rate for the scale parameters.
    #[arg(long, help_heading = "Training options", default_value = "5e-3")]
    pub lr_scale: f64,

    /// Learning rate for the rotation parameters.
    #[arg(long, help_heading = "Training options", default_value = "2e-3")]
    pub lr_rotation: f64,

    /// Max nr. of splats. This is only an upper bound, the actual final number of splats is NOT determined by this.
    #[arg(long, help_heading = "Refine options", default_value = "10000000")]
    pub max_splats: u32,

    /// Frequency of 'refinement' where gaussians are replaced and densified. This should
    /// roughly be the number of images it takes to properly "cover" your scene.
    #[arg(
        long,
        help_heading = "Refine options",
        default_value = "200",
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    pub refine_every: u32,

    /// Threshold to control splat growth. Lower means faster growth.
    #[arg(long, help_heading = "Refine options", default_value = "0.0025")]
    pub growth_grad_threshold: f32,

    /// What fraction of splats that are deemed as needing to grow do actually grow.
    /// Increase this to make splats grow more aggressively.
    #[arg(long, help_heading = "Refine options", default_value = "0.25")]
    pub growth_select_fraction: f32,

    /// Iteration at which splat growth starts. Clamped to `growth_stop_iter`.
    #[arg(long, help_heading = "Refine options", default_value = "0")]
    pub growth_start_iter: u32,

    /// Period after which splat growth stops.
    #[arg(long, help_heading = "Refine options", default_value = "15000")]
    pub growth_stop_iter: u32,

    /// Split any splat whose max screen-space extent exceeds this fraction of
    /// the image dimension, shrinking the children so they land at (at most)
    /// this size on screen. 0 disables.
    #[arg(long, help_heading = "Refine options", default_value = "0.5")]
    pub split_at_screen_size: f32,

    /// Weight of SSIM loss (compared to l1 loss)
    #[clap(long, help_heading = "Training options", default_value = "0.2")]
    pub ssim_weight: f32,

    /// Factor of the opacity decay.
    #[arg(long, help_heading = "Training options", default_value = "0.004")]
    pub opac_decay: f32,

    /// Mip-Splatting 3D-filter strength (the paper's `s`): each splat gets a
    /// per-splat world-space scale floor `sqrt(min_scale_factor) · pixel size
    /// at the nearest observing camera`, i.e. a ~0.32px std-dev floor at the
    /// default. Folded into scales/opacity at render (and baked at export),
    /// never optimized. 0 disables the filter.
    #[arg(long, help_heading = "Training options", default_value = "0.1")]
    pub min_scale_factor: f32,

    /// Weight of l1 loss on alpha if input view has transparency.
    #[arg(long, help_heading = "Refine options", default_value = "0.1")]
    pub match_alpha_weight: f32,

    #[arg(long, help_heading = "Refine options", default_value = "0.0")]
    pub lpips_loss_weight: f32,

    /// Base background color (R,G,B) used during training.
    #[arg(
        long,
        help_heading = "Training options",
        default_value = "0,0,0",
        value_delimiter = ',',
        num_args = 3
    )]
    pub background_color: Vec<f32>,

    /// Strength of random noise added to the background color each step.
    /// Noise is uniform in [-strength, +strength], clamped to [0, 1].
    #[arg(long, help_heading = "Training options", default_value = "0.1")]
    pub background_noise_strength: f32,

    /// Number of LOD levels to generate after initial training (0 = disabled).
    #[arg(long, help_heading = "LOD options", default_value = "0")]
    pub lod_levels: u32,

    /// Number of refinement training steps per LOD level.
    #[arg(long, help_heading = "LOD options", default_value = "5000")]
    pub lod_refine_steps: u32,

    /// Percentage of gaussians to keep at each LOD level (1-100).
    #[arg(long, help_heading = "LOD options", default_value = "50")]
    pub lod_decimation_keep: u32,

    /// Percentage to scale source images at each LOD level (1-100).
    #[arg(long, help_heading = "LOD options", default_value = "50")]
    pub lod_image_scale: u32,

    /// Scene scale used for random splat initialization.
    /// When no init is provided, splats are randomly placed
    /// inside camera frustums up to this depth. By default this is
    /// estimated from the camera spacing (with a 1m minimum).
    #[arg(long, help_heading = "Training options")]
    pub random_init_scene_scale: Option<f32>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self::parse_from([""])
    }
}

impl TrainConfig {
    pub fn total_iters(&self) -> u32 {
        self.total_train_iters + self.lod_levels * self.lod_refine_steps
    }
}
