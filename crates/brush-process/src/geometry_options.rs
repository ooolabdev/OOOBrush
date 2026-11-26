//! Fork-specific geometry CLI options and effective loss resolution.

use brush_train::config::{
    DEPTH_DISTORTION_WEIGHT, FREE_SPACE_WEIGHT, GeometryLossWeights, ORIENTATION_WEIGHT,
};
use clap::Args;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Default, clap::ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GeometrySupervision {
    #[default]
    Auto,
    On,
    Off,
}

#[derive(Clone, Default, Args, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct GeometryOptions {
    /// Default for the three geometry losses. `Auto` detects `depth-calibration.json`; `Off` disables all three.
    #[arg(
        long,
        value_enum,
        default_value = "auto",
        help_heading = "Training options"
    )]
    #[serde(default)]
    pub geometry_supervision: GeometrySupervision,
    /// Override the shortest-axis orientation loss switch (requires depth and normals).
    #[arg(long, action = clap::ArgAction::Set, help_heading = "Geometry options")]
    #[serde(default)]
    pub geometry_orientation: Option<bool>,
    /// Override the training free-space opacity loss switch (requires calibrated free-space maps).
    #[arg(long, action = clap::ArgAction::Set, help_heading = "Geometry options")]
    #[serde(default)]
    pub geometry_free_space: Option<bool>,
    /// Override the ray depth dispersion loss switch (does not require geometry maps).
    #[arg(long, action = clap::ArgAction::Set, help_heading = "Geometry options")]
    #[serde(default)]
    pub geometry_depth_distortion: Option<bool>,
    /// Weight of the enabled orientation loss (default 0.05; zero disables it).
    #[arg(long, visible_alias = "orientation-loss-weight", value_parser = parse_geometry_weight, help_heading = "Geometry options")]
    #[serde(default)]
    pub geometry_orientation_weight: Option<f32>,
    /// Weight of the enabled free-space opacity loss (default 0.02; zero disables it).
    #[arg(long, visible_alias = "free-space-weight", value_parser = parse_geometry_weight, help_heading = "Geometry options")]
    #[serde(default)]
    pub geometry_free_space_weight: Option<f32>,
    /// Weight of the enabled ray depth dispersion loss (default 0.01; zero disables it).
    #[arg(long, visible_alias = "depth-distortion-weight", value_parser = parse_geometry_weight, help_heading = "Geometry options")]
    #[serde(default)]
    pub geometry_depth_distortion_weight: Option<f32>,
}

fn validate_geometry_weight(value: f32) -> Result<f32, String> {
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err("geometry weight must be finite and nonnegative".to_owned())
    }
}

fn parse_geometry_weight(value: &str) -> Result<f32, String> {
    validate_geometry_weight(value.parse::<f32>().map_err(|err| err.to_string())?)
}

impl GeometryOptions {
    /// Resolve each loss independently. A weight alone does not enable a loss;
    /// the explicit switch overrides `Auto`/`On`, and the master `Off` always wins.
    pub fn resolve(&self, prepared: bool) -> anyhow::Result<GeometryLossWeights> {
        let default_enabled = match self.geometry_supervision {
            GeometrySupervision::Auto => prepared,
            GeometrySupervision::On => true,
            GeometrySupervision::Off => false,
        };
        let resolve = |switch: Option<bool>, value: Option<f32>, default_weight: f32| {
            // Also validate deserialized/programmatically supplied values, not just CLI input.
            let weight = validate_geometry_weight(value.unwrap_or(default_weight))
                .map_err(anyhow::Error::msg)?;
            Ok::<_, anyhow::Error>(
                if !matches!(self.geometry_supervision, GeometrySupervision::Off)
                    && switch.unwrap_or(default_enabled)
                {
                    weight
                } else {
                    0.0
                },
            )
        };
        Ok(GeometryLossWeights {
            orientation: resolve(
                self.geometry_orientation,
                self.geometry_orientation_weight,
                ORIENTATION_WEIGHT,
            )?,
            free_space: resolve(
                self.geometry_free_space,
                self.geometry_free_space_weight,
                FREE_SPACE_WEIGHT,
            )?,
            depth_distortion: resolve(
                self.geometry_depth_distortion,
                self.geometry_depth_distortion_weight,
                DEPTH_DISTORTION_WEIGHT,
            )?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args_file::{config_to_args, merge_configs};
    use crate::config::TrainStreamConfig;
    use clap::Parser;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test(unsupported = test)]
    fn extracted_geometry_config_keeps_the_flat_saved_contract() {
        let original = TrainStreamConfig::try_parse_from([
            "brush",
            "--geometry-supervision",
            "on",
            "--geometry-orientation",
            "false",
            "--geometry-free-space-weight",
            "0.04",
            "--geometry-depth-distortion-weight",
            "0.005",
        ])
        .unwrap();
        let value = serde_json::to_value(&original).unwrap();
        assert_eq!(value["geometry-supervision"], "on");
        assert_eq!(value["geometry-orientation"], false);
        assert!(value.get("geometry").is_none());
        assert!(value.get("geometry-free-space-weight").is_some());
        let restored: TrainStreamConfig = serde_json::from_value(value).unwrap();
        assert_eq!(
            restored
                .process_config
                .geometry_loss_weights(false)
                .unwrap(),
            original
                .process_config
                .geometry_loss_weights(false)
                .unwrap(),
        );
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn geometry_switches_and_weights_round_trip_and_cli_overrides_file() {
        let file = TrainStreamConfig::try_parse_from([
            "brush",
            "--seed",
            "123",
            "--geometry-supervision",
            "on",
            "--geometry-orientation",
            "true",
            "--geometry-free-space",
            "false",
            "--geometry-orientation-weight",
            "0.025",
            "--geometry-depth-distortion-weight",
            "0.0025",
        ])
        .unwrap();
        let args = config_to_args(&file).join(" ");
        assert!(args.contains("--geometry-orientation true"));
        assert!(args.contains("--geometry-free-space false"));
        let round_trip = merge_configs(&file, &TrainStreamConfig::default());
        assert_eq!(
            round_trip
                .process_config
                .geometry_loss_weights(false)
                .unwrap(),
            file.process_config.geometry_loss_weights(false).unwrap()
        );
        let cli = TrainStreamConfig::try_parse_from([
            "brush",
            "--geometry-orientation",
            "false",
            "--geometry-free-space",
            "true",
            "--geometry-depth-distortion-weight",
            "0.005",
        ])
        .unwrap();
        let merged = merge_configs(&file, &cli);
        assert_eq!(merged.process_config.seed, 123);
        let weights = merged.process_config.geometry_loss_weights(false).unwrap();
        assert_eq!(weights.orientation, 0.0);
        assert_eq!(weights.free_space, 0.02);
        assert_eq!(weights.depth_distortion, 0.005);
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn geometry_defaults_preserve_prepared_and_rgb_training() {
        let config = TrainStreamConfig::default();
        assert_eq!(
            config.process_config.geometry_loss_weights(false).unwrap(),
            GeometryLossWeights::default()
        );
        assert_eq!(
            config.process_config.geometry_loss_weights(true).unwrap(),
            GeometryLossWeights::standard()
        );
        let explicit_on =
            TrainStreamConfig::try_parse_from(["brush", "--geometry-supervision", "on"]).unwrap();
        assert_eq!(
            explicit_on
                .process_config
                .geometry_loss_weights(false)
                .unwrap(),
            GeometryLossWeights::standard()
        );
        let weight_only =
            TrainStreamConfig::try_parse_from(["brush", "--geometry-orientation-weight", "0.025"])
                .unwrap();
        assert_eq!(
            weight_only
                .process_config
                .geometry_loss_weights(false)
                .unwrap(),
            GeometryLossWeights::default()
        );
        assert_eq!(
            weight_only
                .process_config
                .geometry_loss_weights(true)
                .unwrap()
                .orientation,
            0.025
        );
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn all_eight_geometry_switch_combinations_are_independent() {
        for mask in 0..8 {
            let flags = [(mask & 1) != 0, (mask & 2) != 0, (mask & 4) != 0];
            let config = TrainStreamConfig::try_parse_from([
                "brush",
                "--geometry-supervision",
                "on",
                "--geometry-orientation",
                if flags[0] { "true" } else { "false" },
                "--geometry-free-space",
                if flags[1] { "true" } else { "false" },
                "--geometry-depth-distortion",
                if flags[2] { "true" } else { "false" },
            ])
            .unwrap();
            let weights = config.process_config.geometry_loss_weights(false).unwrap();
            assert_eq!(
                weights.orientation,
                if flags[0] { ORIENTATION_WEIGHT } else { 0.0 }
            );
            assert_eq!(
                weights.free_space,
                if flags[1] { FREE_SPACE_WEIGHT } else { 0.0 }
            );
            assert_eq!(
                weights.depth_distortion,
                if flags[2] {
                    DEPTH_DISTORTION_WEIGHT
                } else {
                    0.0
                }
            );
        }
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn geometry_overrides_and_zero_weights_respect_master_switch() {
        let mut config = TrainStreamConfig::try_parse_from([
            "brush",
            "--geometry-orientation",
            "true",
            "--geometry-orientation-weight",
            "0.025",
            "--geometry-free-space",
            "false",
            "--free-space-weight",
            "0.04",
            "--geometry-depth-distortion",
            "true",
            "--depth-distortion-weight",
            "0.005",
        ])
        .unwrap()
        .process_config;
        let expected = GeometryLossWeights {
            orientation: 0.025,
            free_space: 0.0,
            depth_distortion: 0.005,
        };
        assert_eq!(config.geometry_loss_weights(false).unwrap(), expected);
        assert_eq!(config.geometry_loss_weights(true).unwrap(), expected);
        config.geometry.geometry_orientation_weight = Some(0.0);
        assert_eq!(config.geometry_loss_weights(true).unwrap().orientation, 0.0);
        config.geometry.geometry_supervision = GeometrySupervision::Off;
        assert_eq!(
            config.geometry_loss_weights(true).unwrap(),
            GeometryLossWeights::default()
        );
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn geometry_weights_reject_invalid_values_and_accept_legacy_saved_configs() {
        for value in ["-1", "NaN", "inf", "-inf"] {
            assert!(
                TrainStreamConfig::try_parse_from([
                    "brush",
                    "--geometry-orientation-weight",
                    value
                ])
                .is_err()
            );
        }
        let config = TrainStreamConfig::default();
        let mut value = serde_json::to_value(config).unwrap();
        let object = value.as_object_mut().unwrap();
        object.retain(|key, _| !key.starts_with("geometry-"));
        let old: TrainStreamConfig = serde_json::from_value(value).unwrap();
        assert_eq!(
            old.process_config.geometry_loss_weights(true).unwrap(),
            GeometryLossWeights::standard()
        );
        let mut invalid = old.process_config;
        invalid.geometry.geometry_depth_distortion_weight = Some(f32::NAN);
        assert!(invalid.geometry_loss_weights(false).is_err());
    }
}
