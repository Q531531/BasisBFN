import torch

from sgbformer.models.bfn import (
    DegradationLatentBFN,
    ResidualBayesianFlow,
    degradation_latent_targets,
    local_entropy_map,
)


def test_accuracy_schedule_telescopes() -> None:
    flow = ResidualBayesianFlow(min_variance=1e-3)
    steps = 10
    alphas = torch.stack(
        [flow.step_accuracy(index, steps) for index in range(1, steps + 1)]
    )
    expected = flow.cumulative_accuracy(torch.tensor(1.0))
    torch.testing.assert_close(alphas.sum(), expected)


def test_linear_and_equal_accuracy_schedules_balance_steps() -> None:
    for schedule in ("linear_beta", "equal_alpha"):
        flow = ResidualBayesianFlow(
            min_variance=0.01,
            accuracy_schedule=schedule,
        )
        alphas = torch.stack([
            flow.step_accuracy(index, 4)
            for index in range(1, 5)
        ])
        torch.testing.assert_close(
            alphas,
            torch.full_like(alphas, 99.0 / 4),
        )
        torch.testing.assert_close(
            alphas.sum(),
            flow.cumulative_accuracy(torch.tensor(1.0)),
        )


def test_sender_state_and_bayesian_update() -> None:
    flow = ResidualBayesianFlow(min_variance=0.01)
    residual = torch.ones(2, 3, 4, 4)
    time = torch.tensor([0.0, 0.5])
    scale = torch.ones(2, 1, 4, 4)
    mean, precision = flow.sender_state(
        residual, time, scale, noise=torch.zeros_like(residual)
    )
    torch.testing.assert_close(mean[0], torch.zeros_like(mean[0]))
    torch.testing.assert_close(precision[0], torch.ones_like(precision[0]))
    expected_beta = flow.cumulative_accuracy(time[1])
    torch.testing.assert_close(
        mean[1], torch.full_like(mean[1], expected_beta / (1 + expected_beta))
    )

    updated_mean, updated_precision = flow.bayesian_update(
        torch.zeros_like(residual),
        torch.ones(2, 1, 4, 4),
        torch.ones_like(residual),
        torch.ones(2, 1, 4, 4),
    )
    torch.testing.assert_close(updated_mean, torch.full_like(residual, 0.5))
    torch.testing.assert_close(
        updated_precision, torch.full_like(updated_precision, 2.0)
    )


def test_entropy_map_is_normalized() -> None:
    constant = torch.full((1, 3, 12, 12), 0.5)
    entropy = local_entropy_map(constant, kernel_size=3, bins=8)
    assert entropy.shape == (1, 1, 12, 12)
    assert float(entropy.min()) >= 0.0
    assert float(entropy.max()) <= 1.0
    torch.testing.assert_close(entropy, torch.zeros_like(entropy))


def test_degradation_latent_flow_shapes_targets_and_gradients() -> None:
    flow = DegradationLatentBFN(
        feature_channels=16,
        global_dim=8,
        spatial_dim=2,
        hidden_channels=4,
        num_categories=3,
        spatial_downsample=2,
    )
    feature = torch.rand(2, 16, 8, 8)
    first = flow(feature, steps=4, return_trajectory=True)
    second = flow(feature, steps=4, return_trajectory=True)
    for name in (
        "global_mean",
        "global_precision",
        "spatial_mean",
        "spatial_precision",
        "category_logits",
    ):
        torch.testing.assert_close(first[name], second[name], atol=0, rtol=0)
    assert first["global_mean"].shape == (2, 8)
    assert first["spatial_mean"].shape == (2, 2, 4, 4)
    assert len(first["trajectory"]) == 4

    degraded = torch.rand(2, 3, 32, 32)
    target = torch.rand_like(degraded)
    global_target, spatial_target = degradation_latent_targets(
        degraded,
        target,
        global_dim=8,
        spatial_dim=2,
        spatial_size=(4, 4),
        category=torch.tensor([0, 2]),
        num_categories=3,
    )
    assert global_target.shape == (2, 8)
    assert spatial_target.shape == (2, 2, 4, 4)
    loss = flow.latent_loss(first, global_target, spatial_target)
    loss.backward()
    assert flow.global_observer[-1].weight.grad is not None
    assert flow.spatial_observer[-1].weight.grad is not None


def test_final_mean_latent_loss_does_not_supervise_each_observation() -> None:
    flow = DegradationLatentBFN(
        feature_channels=4,
        global_dim=8,
        spatial_dim=2,
        hidden_channels=4,
        num_categories=3,
        loss_mode="final_mean",
        accuracy_schedule="linear_beta",
    )
    global_mean = torch.zeros(1, 8, requires_grad=True)
    spatial_mean = torch.zeros(1, 2, 2, 2, requires_grad=True)
    global_observation = torch.zeros(1, 8, requires_grad=True)
    spatial_observation = torch.zeros(1, 2, 2, 2, requires_grad=True)
    outputs = {
        "global_mean": global_mean,
        "spatial_mean": spatial_mean,
        "global_observations": [global_observation],
        "spatial_observations": [spatial_observation],
    }
    loss = flow.latent_loss(
        outputs,
        torch.ones_like(global_mean),
        torch.ones_like(spatial_mean),
    )
    loss.backward()
    assert global_mean.grad is not None
    assert spatial_mean.grad is not None
    assert global_observation.grad is None
    assert spatial_observation.grad is None


def test_matched_update_modes_share_parameters_and_adaptive_precision() -> None:
    kwargs = {
        "feature_channels": 4,
        "global_dim": 8,
        "spatial_dim": 2,
        "hidden_channels": 4,
        "num_categories": 3,
        "accuracy_schedule": "linear_beta",
        "loss_mode": "final_mean",
        "loss_weighting": "adaptive_accuracy",
    }
    bayesian = DegradationLatentBFN(**kwargs, update_mode="bayesian")
    deterministic = DegradationLatentBFN(
        **kwargs,
        update_mode="deterministic",
    )
    deterministic.load_state_dict(bayesian.state_dict(), strict=True)
    assert {
        name: parameter.numel()
        for name, parameter in bayesian.named_parameters()
    } == {
        name: parameter.numel()
        for name, parameter in deterministic.named_parameters()
    }

    feature = torch.rand(2, 4, 8, 8)
    outputs_b = bayesian(feature, steps=4)
    outputs_d = deterministic(feature, steps=4)
    for name in (
        "global_accuracies",
        "spatial_accuracies",
    ):
        values_b = outputs_b[name]
        values_d = outputs_d[name]
        assert isinstance(values_b, list)
        assert isinstance(values_d, list)
        # The first head receives exactly the same state. Later states differ
        # by design because the update semantics are the controlled variable.
        torch.testing.assert_close(values_b[0], values_d[0], atol=0, rtol=0)

    for outputs in (outputs_b, outputs_d):
        expected_global = 1 + torch.stack(
            outputs["global_accuracies"]
        ).sum(dim=0)
        expected_spatial = 1 + torch.stack(
            outputs["spatial_accuracies"]
        ).sum(dim=0)
        torch.testing.assert_close(
            outputs["global_precision"],
            expected_global,
        )
        torch.testing.assert_close(
            outputs["spatial_precision"],
            expected_spatial,
        )
    assert not torch.equal(outputs_b["global_mean"], outputs_d["global_mean"])


def test_both_update_modes_are_train_eval_identical() -> None:
    feature = torch.rand(2, 4, 8, 8)
    for update_mode in ("deterministic", "bayesian"):
        flow = DegradationLatentBFN(
            feature_channels=4,
            global_dim=8,
            spatial_dim=2,
            hidden_channels=4,
            num_categories=3,
            accuracy_schedule="linear_beta",
            loss_mode="final_mean",
            update_mode=update_mode,
            loss_weighting="adaptive_accuracy",
        )
        flow.train()
        training = flow(feature, steps=4, return_trajectory=True)
        flow.eval()
        inference = flow(feature, steps=4, return_trajectory=True)
        for name in (
            "global_mean",
            "global_precision",
            "spatial_mean",
            "spatial_precision",
        ):
            torch.testing.assert_close(
                training[name],
                inference[name],
                atol=0,
                rtol=0,
            )
        for training_step, inference_step in zip(
            training["trajectory"],
            inference["trajectory"],
        ):
            torch.testing.assert_close(
                training_step,
                inference_step,
                atol=0,
                rtol=0,
            )


def test_adaptive_latent_loss_uses_detached_accumulated_accuracy() -> None:
    flow = DegradationLatentBFN(
        feature_channels=4,
        global_dim=8,
        spatial_dim=2,
        hidden_channels=4,
        num_categories=3,
        loss_mode="final_mean",
        loss_weighting="adaptive_accuracy",
    )
    global_mean = torch.zeros(1, 8, requires_grad=True)
    spatial_mean = torch.zeros(1, 2, 2, 2, requires_grad=True)
    global_precision = torch.arange(
        2,
        10,
        dtype=torch.float32,
    ).view(1, 8).requires_grad_()
    spatial_precision = torch.full(
        (1, 2, 2, 2),
        3.0,
        requires_grad=True,
    )
    outputs = {
        "global_mean": global_mean,
        "global_precision": global_precision,
        "spatial_mean": spatial_mean,
        "spatial_precision": spatial_precision,
        "global_observations": [],
        "spatial_observations": [],
    }
    loss = flow.latent_loss(
        outputs,
        torch.ones_like(global_mean),
        torch.ones_like(spatial_mean),
    )
    loss.backward()
    assert global_mean.grad is not None
    assert spatial_mean.grad is not None
    assert global_precision.grad is None
    assert spatial_precision.grad is None
    global_weight, spatial_weight = flow.latent_loss_weights(outputs)
    torch.testing.assert_close(global_weight, global_precision.detach() - 1)
    torch.testing.assert_close(spatial_weight, spatial_precision.detach() - 1)
