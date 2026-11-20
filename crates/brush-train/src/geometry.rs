use brush_dataset::scene::{GEOMETRY_STRIDE, GeometryMaps};
use brush_render::camera::Camera;
use burn::tensor::{Bool, Tensor, s};
use glam::UVec2;

/// Render raw moments with the same visibility, footprint, and transmittance as RGB.
/// Use DC features in a separate experimental pass to reuse the full compositing VJP.
pub(crate) async fn depth_distortion(
    mut splats: brush_render::gaussian_splats::Splats,
    camera: &Camera,
    size: UVec2,
) -> Tensor<1> {
    let means = splats.means();
    let device = means.device();
    let vector = |v: glam::Vec3| Tensor::<1>::from_floats(v.to_array(), &device).reshape([1, 3]);
    let z =
        ((means - vector(camera.position)) * vector(camera.rotation * glam::Vec3::Z)).sum_dim(1);
    // RGB is nonnegative. Encode the signed first moment in two channels.
    let x = z.clamp_min(1e-6).log() / 16.0;
    let features = Tensor::cat(
        vec![
            x.clone().clamp_min(0.),
            x.clone().neg().clamp_min(0.),
            x.square(),
        ],
        1,
    );
    let sh = ((features - 0.5) / brush_render::shaders::SH_C0).unsqueeze_dim(1);
    splats.sh_coeffs = burn::module::Param::initialized(splats.sh_coeffs.id, sh);
    let img = brush_render::bwd::render_splats(splats, camera, size, glam::Vec3::ZERO)
        .await
        .img;
    let first = img.clone().slice(s![.., .., 0..1]) - img.clone().slice(s![.., .., 1..2]);
    let second = img.clone().slice(s![.., .., 2..3]);
    let opacity = img.slice(s![.., .., 3..4]);
    ((opacity * second - first.square()) * 256.)
        .clamp_min(0.)
        .mean()
}

/// Black background and binary DC colours give accumulated foreground opacity at three planes.
pub(crate) async fn free_space_opacity(
    mut splats: brush_render::gaussian_splats::Splats,
    camera: &Camera,
    size: UVec2,
    planes: [f32; 3],
    channels: burn::tensor::TensorData,
) -> Tensor<1> {
    use burn::module::Param;
    splats.transforms = Param::initialized(splats.transforms.id, splats.transforms.val().detach());
    let means = splats.means();
    let device = means.device();
    let vector = |v: glam::Vec3| Tensor::<1>::from_floats(v.to_array(), &device).reshape([1, 3]);
    let z =
        ((means - vector(camera.position)) * vector(camera.rotation * glam::Vec3::Z)).sum_dim(1);
    let features = Tensor::cat(
        planes.map(|cut| z.clone().lower_elem(cut).float()).to_vec(),
        1,
    );
    let dc = ((features - 0.5) / brush_render::shaders::SH_C0).unsqueeze_dim(1);
    splats.sh_coeffs = Param::initialized(splats.sh_coeffs.id, dc);
    let img = brush_render::bwd::render_splats(splats, camera, size, glam::Vec3::ZERO)
        .await
        .img;
    let stride = GEOMETRY_STRIDE as isize;
    let alpha = img.slice(s![..;stride, ..;stride, 0..3]);
    let channels = Tensor::<2>::from_data(channels, &device);
    let weights = Tensor::stack::<3>(
        (1..=3)
            .map(|i| channels.clone().equal_elem(i as f32).float())
            .collect(),
        2,
    );
    (alpha * weights.clone()).sum() / weights.sum().clamp_min(1.0)
}

/// Reject silhouettes and depth discontinuities before taking central differences.
fn neighbourhood(z: &Tensor<2>, valid: &Tensor<2, Bool>) -> Tensor<2, Bool> {
    let [h, w] = z.dims();
    let centre = z.clone().slice(s![1..h - 1, 1..w - 1]);
    let mut mask = valid.clone().slice(s![1..h - 1, 1..w - 1]);
    for (y, x) in [(0, 1), (2, 1), (1, 0), (1, 2)] {
        let neighbour = z.clone().slice(s![y..y + h - 2, x..x + w - 2]);
        let delta = (neighbour - centre.clone()).abs();
        mask = mask
            .bool_and(valid.clone().slice(s![y..y + h - 2, x..x + w - 2]))
            .bool_and(delta.lower_equal(centre.clone() * 0.05));
    }
    mask
}

/// Align the blended shortest Gaussian axes in valid depth-map interiors.
pub(crate) fn orientation_loss(img: Tensor<3>, targets: GeometryMaps) -> Tensor<1> {
    let stride = GEOMETRY_STRIDE as isize;
    let device = img.device();
    let sampled = img.slice(s![..;stride, ..;stride, ..]);
    let [h, w, _] = sampled.dims();
    if h < 3 || w < 3 {
        return Tensor::zeros([1], &device);
    }
    let depth = Tensor::<2>::from_data(targets.depth.expect("geometry depth mask"), &device);
    let valid = sampled
        .clone()
        .slice(s![.., .., 3..4])
        .squeeze_dim::<2>(2)
        .greater_elem(0.3)
        .bool_and(
            sampled
                .clone()
                .slice(s![.., .., 4..5])
                .squeeze_dim::<2>(2)
                .greater_elem(0.0),
        )
        .bool_and(depth.clone().greater_elem(0.0));
    let target = Tensor::<3>::from_data(targets.normals.expect("geometry normals"), &device)
        .slice(s![1..h - 1, 1..w - 1, ..]);
    let len = target.clone().square().sum_dim(2);
    let mask =
        neighbourhood(&depth, &valid).bool_and(len.clone().squeeze_dim::<2>(2).greater_elem(0.25));
    let target = target / len.clamp_min(1e-12).sqrt();
    let n = sampled.slice(s![1..h - 1, 1..w - 1, 5..8]);
    let magnitude = n.clone().square().sum_dim(2).clamp_min(1e-12).sqrt();
    let strength = magnitude.clone().detach().squeeze_dim::<2>(2);
    let support = mask.bool_and(strength.clone().greater_elem(0.05)).float();
    let cosine = (n / magnitude * target)
        .sum_dim(2)
        .squeeze_dim::<2>(2)
        .clamp(-1., 1.);
    ((cosine.neg() + 1.) * strength.clamp(0., 1.) * support.clone()).sum()
        / support.sum().clamp_min(1.)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn orientation_uses_depth_only_as_an_edge_mask() {
        use burn::tensor::TensorData;
        let device = brush_cube::test_helpers::test_device().await.into();
        let image = [0.0_f32, 0., 0., 1., 2., 1., 0., 0.].repeat(12 * 12);
        let image = Tensor::<1>::from_floats(image.as_slice(), &device).reshape([12, 12, 8]);
        for (scale, centre, expected) in [(0.3_f32, 0.3, 1.), (0.8, 0.8, 1.), (0.8, 0.2, 0.)] {
            let mut depth = vec![scale; 9];
            depth[4] = centre;
            let targets = GeometryMaps {
                depth: Some(TensorData::new(depth, [3, 3])),
                normals: Some(TensorData::new([0_f32, 1., 0.].repeat(9), [3, 3, 3])),
                ..Default::default()
            };
            let loss = orientation_loss(image.clone(), targets)
                .into_scalar_async::<f32>()
                .await
                .unwrap();
            assert!((loss - expected).abs() < 1e-5);
        }
    }

    #[tokio::test]
    async fn distortion_separates_depth_layers_and_differentiates_opacity() {
        use brush_render::gaussian_splats::{SplatRenderMode, Splats};
        use burn::module::Module;
        let device = brush_cube::test_helpers::test_device().await.into();
        let camera = Camera::new(
            glam::Vec3::ZERO,
            glam::Quat::IDENTITY,
            1.,
            1.,
            glam::vec2(0.5, 0.5),
            Default::default(),
        );
        let size = UVec2::splat(21);
        let scene = |far, opacity| {
            Splats::from_raw(
                vec![0., 0., 1., 0., 0., far],
                [1., 0., 0., 0.].repeat(2),
                vec![-1.; 6],
                vec![0.; 6],
                vec![-1., opacity],
                SplatRenderMode::Default,
                &device,
            )
            .train()
        };
        let value = async |far, opacity| {
            depth_distortion(scene(far, opacity), &camera, size)
                .await
                .inner()
                .into_scalar_async::<f32>()
                .await
                .unwrap()
        };
        assert!(value(1., 0.).await.abs() < 1e-5);
        let model = scene(2., 0.);
        let loss = depth_distortion(model.clone(), &camera, size).await;
        assert!(
            loss.clone()
                .inner()
                .into_scalar_async::<f32>()
                .await
                .unwrap()
                > 1e-4
        );
        let mut grads = loss.backward();
        let g = model
            .raw_opacities
            .val()
            .grad_remove(&mut grads)
            .unwrap()
            .into_data_async()
            .await
            .unwrap()
            .try_into_vec::<f32>()
            .unwrap();
        let fd = (value(2., 0.01).await - value(2., -0.01).await) / 0.02;
        assert!(g[1] > 0. && (g[1] - fd).abs() < 0.02 * fd.abs() + 1e-5);
        let centers = model
            .transforms
            .val()
            .grad_remove(&mut grads)
            .unwrap()
            .into_data_async()
            .await
            .unwrap()
            .try_into_vec::<f32>()
            .unwrap();
        let dz = (value(2.005, 0.).await - value(1.995, 0.).await) / 0.01;
        assert!(
            (centers[12] - dz).abs() < 0.05 * dz.abs() + 2e-5,
            "Z derivative {}, expected {}",
            centers[12],
            dz
        );
    }
    #[tokio::test]
    async fn free_space_only_penalizes_opacity_in_front_of_selected_plane() {
        use brush_render::gaussian_splats::{SplatRenderMode, Splats};
        use burn::module::Module;
        let device = brush_cube::test_helpers::test_device().await.into();
        let camera = Camera::new(
            glam::Vec3::ZERO,
            glam::Quat::IDENTITY,
            1.,
            1.,
            glam::vec2(0.5, 0.5),
            Default::default(),
        );
        let model = Splats::from_raw(
            vec![0., 0., 1., 0., 0., 2.],
            [1., 0., 0., 0.].repeat(2),
            vec![-1.; 6],
            vec![0.; 6],
            vec![0.; 2],
            SplatRenderMode::Default,
            &device,
        )
        .train();
        let loss = async |channel: f32| {
            let labels = burn::tensor::TensorData::new(vec![channel; 36], [6, 6]);
            free_space_opacity(
                model.clone(),
                &camera,
                UVec2::splat(21),
                [0.5, 1.5, 3.],
                labels,
            )
            .await
        };
        for c in [0., 1.] {
            assert!(
                loss(c)
                    .await
                    .inner()
                    .into_scalar_async::<f32>()
                    .await
                    .unwrap()
                    .abs()
                    < 1e-6
            );
        }
        let front = loss(2.).await;
        let value = front
            .clone()
            .inner()
            .into_scalar_async::<f32>()
            .await
            .unwrap();
        assert!(
            value > 0.
                && loss(3.)
                    .await
                    .inner()
                    .into_scalar_async::<f32>()
                    .await
                    .unwrap()
                    > value
        );
        let mut grads = front.backward();
        assert!(model.transforms.val().grad_remove(&mut grads).is_none());
        assert!(model.sh_coeffs.val().grad_remove(&mut grads).is_none());
        let opacity = model
            .raw_opacities
            .val()
            .grad_remove(&mut grads)
            .unwrap()
            .into_data_async()
            .await
            .unwrap()
            .try_into_vec::<f32>()
            .unwrap();
        assert!(opacity[0] > 0. && opacity[1].abs() < 1e-6, "{opacity:?}");
    }
}
