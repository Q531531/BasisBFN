import pytest
import torch

from sgbformer import BasisBFN, BasisBFNLoss, SGBFormer, SGBFormerLoss
from sgbformer.models import SpectralGate, spectral_scale_delta


def tiny_model(bfn_overrides=None) -> SGBFormer:
    bfn = {
        "steps": 2,
        "min_variance": 0.01,
        "base_channels": 8,
        "channel_multipliers": (1, 2),
        "blocks_per_level": 1,
        "attention_heads": 2,
        "entropy_kernel": 3,
        "entropy_bins": 4,
        "entropy_threshold": 0.0,
    }
    bfn.update(bfn_overrides or {})
    return SGBFormer(
        dim=8,
        num_blocks=(1, 1, 1, 1),
        refinement_blocks=1,
        heads=(1, 2, 4, 8),
        ffn_expansion_factor=2.0,
        semantic={
            "encoder": "none",
            "embedding_dim": 16,
            "context_tokens": 2,
        },
        bfn=bfn,
    )


def test_public_basisbfn_names_keep_checkpoint_compatibility() -> None:
    assert SGBFormer is BasisBFN
    assert SGBFormerLoss is BasisBFNLoss


def test_spectral_gate_shape_and_gradient() -> None:
    layer = SpectralGate(8)
    image = torch.randn(2, 8, 15, 17, requires_grad=True)
    output, gate = layer(image, return_gate=True)
    assert output.shape == image.shape
    assert gate.shape == (2, 8, 15, 9)
    output.mean().backward()
    assert image.grad is not None
    assert layer.gate.weight.grad is not None


def test_spectral_gate_can_preserve_a_pretrained_foundation_at_init() -> None:
    layer = SpectralGate(4, initial_gate=0.999)
    image = torch.randn(1, 4, 13, 15)
    output, gate = layer(image, return_gate=True)
    torch.testing.assert_close(
        gate,
        torch.full_like(gate, 0.999),
        atol=1e-7,
        rtol=0,
    )
    torch.testing.assert_close(output, image * 0.999, atol=2e-6, rtol=1e-6)
    with pytest.raises(ValueError, match="initial_gate"):
        SpectralGate(4, initial_gate=1.0)


def test_zero_spectral_scale_delta_is_exactly_zero() -> None:
    feature = torch.randn(2, 4, 13, 15)
    scale_delta = torch.zeros(2, 4, 13, 8)
    delta = spectral_scale_delta(feature, scale_delta)
    torch.testing.assert_close(delta, torch.zeros_like(feature), atol=0, rtol=0)
    with pytest.raises(ValueError, match="real FFT shape"):
        spectral_scale_delta(feature, torch.zeros(2, 4, 13, 15))


def test_training_and_restoration_with_odd_size() -> None:
    torch.manual_seed(3)
    model = tiny_model()
    degraded = torch.rand(1, 3, 17, 19)
    target = torch.rand_like(degraded)
    outputs = model(degraded, target)
    assert outputs["restored"].shape == degraded.shape
    assert outputs["sampling_mask"].shape == (1, 1, 17, 19)

    loss, parts = SGBFormerLoss(bfn_weight=0.01)(outputs, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(value) for value in parts.values())

    model.eval()
    first = model.restore(degraded, deterministic=True)
    second = model.restore(degraded, deterministic=True)
    torch.testing.assert_close(first, second)
    assert float(first.min()) >= 0.0
    assert float(first.max()) <= 1.0


def test_observation_rich_receiver_uses_degraded_image_in_all_paths() -> None:
    model = tiny_model({
        "condition_mode": "observation_rich",
        "training_final_state": "rollout",
        "training_rollout_steps": 2,
        "rollout_batch_fraction": 1.0,
    })
    assert model.receiver.observation_channels == 6
    degraded = torch.rand(2, 3, 17, 19)
    target = torch.rand_like(degraded)
    outputs = model(degraded, target, training_stage="joint")
    outputs["bfn_loss"].backward()
    assert model.receiver.input_projection.weight.grad is not None
    restored = model.restore(degraded, deterministic=True)
    assert restored.shape == degraded.shape
    direct = tiny_model({
        "condition_mode": "observation_rich",
        "inference_mode": "direct",
    })
    assert sum(p.numel() for p in model.receiver.parameters()) == sum(
        p.numel() for p in direct.receiver.parameters()
    )
    direct_outputs = direct(
        degraded,
        target,
        training_stage="residual",
    )
    direct_outputs["restored"].mean().backward()
    assert direct.receiver.input_projection.weight.grad is not None
    assert direct.restore(degraded, deterministic=True).shape == degraded.shape


def test_residual_scale_is_applied_only_at_the_image_boundary() -> None:
    class ConstantReceiver(torch.nn.Module):
        def forward(
            self,
            belief_mean,
            uncertainty,
            time,
            structural,
            decoder_feature,
        ):
            del uncertainty, time, structural, decoder_feature
            return torch.ones_like(belief_mean)

    model = tiny_model({"residual_scale": 0.1})
    model.receiver = ConstantReceiver()
    details = model.restore(
        torch.zeros(1, 3, 16, 16),
        deterministic=True,
        return_details=True,
    )
    torch.testing.assert_close(
        details["residual"],
        torch.full_like(details["residual"], 0.1),
    )
    zero_gain = model.restore(
        torch.zeros(1, 3, 16, 16),
        deterministic=True,
        residual_gain=0.0,
        return_details=True,
    )
    torch.testing.assert_close(
        zero_gain["residual"],
        torch.zeros_like(zero_gain["residual"]),
    )
    torch.testing.assert_close(
        zero_gain["restored"],
        zero_gain["structural"],
    )
    with pytest.raises(ValueError, match="residual_gain"):
        model.restore(torch.zeros(1, 3, 8, 8), residual_gain=1.1)

    bounded = tiny_model({
        "residual_scale": 0.1,
        "receiver_output_mode": "tanh",
    })
    bounded.receiver = ConstantReceiver()
    bounded_details = bounded.restore(
        torch.zeros(1, 3, 16, 16),
        deterministic=True,
        return_details=True,
    )
    assert float(bounded_details["residual"].abs().max()) < 0.1


def test_registered_residual_state_updates_are_parameter_free_and_distinct() -> None:
    mean = torch.ones(1, 3, 2, 2)
    precision = torch.full((1, 1, 2, 2), 2.0)
    observation = torch.full_like(mean, 3.0)
    accuracy = torch.full_like(precision, 4.0)

    deterministic = tiny_model({"update_mode": "deterministic"})
    bayesian = tiny_model({"update_mode": "bayesian"})
    deterministic_state = deterministic.update_residual_state(
        mean,
        precision,
        observation,
        accuracy,
    )
    bayesian_state = bayesian.update_residual_state(
        mean,
        precision,
        observation,
        accuracy,
    )

    torch.testing.assert_close(
        deterministic_state[0],
        mean + accuracy / (precision + accuracy) * observation,
    )
    torch.testing.assert_close(
        bayesian_state[0],
        (precision * mean + accuracy * observation) / (precision + accuracy),
    )
    torch.testing.assert_close(
        deterministic_state[1],
        bayesian_state[1],
    )
    assert not torch.equal(deterministic_state[0], bayesian_state[0])

    neutral_mean = torch.zeros_like(mean)
    deterministic_first = deterministic.update_residual_state(
        neutral_mean,
        torch.ones_like(precision),
        observation,
        accuracy,
    )
    bayesian_first = bayesian.update_residual_state(
        neutral_mean,
        torch.ones_like(precision),
        observation,
        accuracy,
    )
    torch.testing.assert_close(deterministic_first[0], bayesian_first[0])


def test_posterior_mean_readout_is_identity_safe_and_uses_exactly_k_calls() -> None:
    model = tiny_model({
        "steps": 2,
        "accuracy_schedule": "linear_beta",
        "mask_mode": "none",
        "update_mode": "bayesian",
        "readout_mode": "posterior_mean",
        "training_final_state": "rollout",
        "training_rollout_steps": 2,
        "rollout_backprop_steps": 1,
        "residual_base_clamp": True,
    })
    image = torch.rand(1, 3, 17, 19)
    details = model.restore(
        image,
        deterministic=True,
        return_details=True,
    )
    baseline = model.restore_structural(image)

    torch.testing.assert_close(details["restored"], baseline, atol=0, rtol=0)
    torch.testing.assert_close(
        details["residual"],
        torch.zeros_like(details["residual"]),
        atol=0,
        rtol=0,
    )
    assert details["receiver_calls"] == 2
    assert details["terminal_readout_time"] is None
    assert details["residual_update_mode"] == "bayesian"
    assert details["residual_readout_mode"] == "posterior_mean"


def test_posterior_mean_training_has_no_terminal_receiver_call() -> None:
    class RecordingReceiver(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(
            self,
            belief_mean,
            uncertainty,
            time,
            structural,
            decoder_feature,
        ):
            del uncertainty, time, structural, decoder_feature
            self.calls += 1
            return torch.ones_like(belief_mean)

    model = tiny_model({
        "steps": 2,
        "update_mode": "deterministic",
        "readout_mode": "posterior_mean",
        "training_final_state": "rollout",
        "training_rollout_steps": 2,
        "rollout_backprop_steps": 1,
        "rollout_batch_fraction": 1.0,
    })
    receiver = RecordingReceiver()
    model.receiver = receiver
    image = torch.rand(1, 3, 16, 16)
    outputs = model(image, image, training_stage="joint")

    # One random-step BFN-loss call plus two rollout calls; no terminal readout.
    assert receiver.calls == 3
    assert outputs["restored"].shape == image.shape


def test_joint_training_supervises_the_terminal_bfn_state() -> None:
    class RecordingReceiver(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.times = []

        def forward(
            self,
            belief_mean,
            uncertainty,
            time,
            structural,
            decoder_feature,
        ):
            del uncertainty, structural, decoder_feature
            self.times.append(time.detach().clone())
            return torch.zeros_like(belief_mean)

    model = tiny_model({
        "residual_scale": 0.1,
        "training_final_state": "terminal",
    })
    receiver = RecordingReceiver()
    model.receiver = receiver
    image = torch.rand(2, 3, 16, 16)
    model(image, image, training_stage="joint")
    assert len(receiver.times) == 2
    torch.testing.assert_close(
        receiver.times[-1],
        torch.ones_like(receiver.times[-1]),
    )


def test_joint_rollout_uses_inference_states_for_a_sub_batch() -> None:
    class RecordingReceiver(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_sizes = []

        def forward(
            self,
            belief_mean,
            uncertainty,
            time,
            structural,
            decoder_feature,
        ):
            del uncertainty, time, structural, decoder_feature
            self.batch_sizes.append(belief_mean.shape[0])
            return torch.zeros_like(belief_mean)

    model = tiny_model({
        "training_final_state": "rollout",
        "training_rollout_steps": 2,
        "rollout_batch_fraction": 0.25,
    })
    receiver = RecordingReceiver()
    model.receiver = receiver
    image = torch.rand(4, 3, 16, 16)
    outputs = model(image, image, training_stage="joint")
    assert receiver.batch_sizes == [4, 1, 1, 1]
    assert outputs["final_loss_mask"].sum() == 1


def test_truncated_rollout_backpropagates_only_through_requested_steps() -> None:
    class WeightedReceiver(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = torch.nn.ParameterList(
                [torch.nn.Parameter(torch.ones(())) for _ in range(4)]
            )
            self.call_index = 0

        def forward(
            self,
            belief_mean,
            uncertainty,
            time,
            structural,
            decoder_feature,
        ):
            del uncertainty, time, structural, decoder_feature
            weight = self.weights[self.call_index]
            self.call_index += 1
            return belief_mean + torch.ones_like(belief_mean) * weight

    model = tiny_model({
        "training_final_state": "rollout",
        "training_rollout_steps": 2,
        "rollout_backprop_steps": 1,
        "rollout_batch_fraction": 1.0,
    })
    receiver = WeightedReceiver()
    model.receiver = receiver
    image = torch.rand(1, 3, 16, 16)
    outputs = model(image, image, training_stage="joint")
    outputs["restored"].sum().backward()
    assert receiver.weights[0].grad is None  # Random Equation (11) state.
    assert receiver.weights[1].grad is None  # Truncated rollout prefix.
    assert receiver.weights[2].grad is not None  # Last rollout update.
    assert receiver.weights[3].grad is not None  # Final receiver call.


def test_paper_backbone_parameter_scale() -> None:
    model = SGBFormer(
        dim=32,
        semantic={"encoder": "none", "embedding_dim": 512},
    )
    backbone_parameters = sum(
        parameter.numel()
        for parameter in model.backbone.parameters()
        if parameter.requires_grad
    )
    assert 12_500_000 < backbone_parameters < 13_500_000


def test_training_stages_isolate_gradients() -> None:
    degraded = torch.rand(1, 3, 16, 16)
    target = torch.rand_like(degraded)

    structure_model = tiny_model()
    structure_output = structure_model(
        degraded, target, training_stage="structure"
    )
    structure_output["structural"].mean().backward()
    assert any(
        parameter.grad is not None
        for parameter in structure_model.backbone.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in structure_model.receiver.parameters()
    )

    bfn_model = tiny_model()
    bfn_output = bfn_model(degraded, target, training_stage="bfn")
    bfn_output["bfn_loss"].backward()
    assert all(
        parameter.grad is None for parameter in bfn_model.backbone.parameters()
    )
    assert any(
        parameter.grad is not None for parameter in bfn_model.receiver.parameters()
    )
