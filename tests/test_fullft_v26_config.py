"""Contract tests for the v26 full fine-tuning + BFN configs.

``load_config`` only performs ``os.path.expandvars``
(sgbformer/config.py:17-19), so unset placeholders such as ``${SSR_REPO}``
survive verbatim and no environment fixture is required, exactly like
``tests/test_config.py``.
"""

from pathlib import Path

import pytest

from sgbformer.config import load_config
from sgbformer.training import cosine_warmup_factor, resolve_training_stages

CONFIG_NAMES = (
    "ssr_bfn_fullft_v26",
    "ssr_bfn_fullft_v26_pilot",
    "ssr_bfn_fullft_v26b",
    "ssr_bfn_fullft_v26b_proxy",
    "ssr_bfn_fullft_v26c_nobalance_proxy",
    "ssr_bfn_fullft_v26d_lrlow_proxy",
)

BALANCED_CONFIG_NAMES = tuple(
    name
    for name in CONFIG_NAMES
    if name != "ssr_bfn_fullft_v26c_nobalance_proxy"
)

# 【已测量】raw (unweighted) loss components from the v26 pilot epoch-2 log:
# loss=0.00201 struct=0.00093 bfn=0.04451 freq=0.00944 latent_cls=0.01701.
# losses.py:184-194 reports these unweighted, losses.py:174-183 sums them with
# the configured weights, so the objective composition is reproducible here.
PILOT_RAW_COMPONENTS = {
    "structural": 0.00093,
    "bfn": 0.04451,
    "frequency": 0.00944,
    "latent_category": 0.01701,
}

EXPECTED_SAMPLING = {
    "mode": "category_balanced",
    "category_patterns": {
        "snow": r".*\.jpe?g",
        "outdoor_rain": r"im_\d+_s\d+_a0[456]\.png",
        "raindrop": r"\d+_rain\.png",
    },
    "target_fractions": {
        "snow": 0.50,
        "outdoor_rain": 0.25,
        "raindrop": 0.25,
    },
}


def load_experiment(name: str) -> dict:
    root = Path(__file__).resolve().parents[1] / "configs" / "experiments"
    return load_config(root / f"{name}.yaml")


@pytest.fixture(scope="module")
def configs() -> dict[str, dict]:
    return {name: load_experiment(name) for name in CONFIG_NAMES}


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_protocol_name_is_new_and_never_frozen_main(name, configs) -> None:
    config = configs[name]
    protocol = config["protocol"]
    assert protocol["name"] == "fullft_bfn_v26"
    assert protocol["name"] != "frozen_main_db"
    # A tripwire block would only be honoured by a budget-locked protocol.
    assert protocol.get("tripwire") is None
    assert config["seed"] == 42


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_foundation_is_fully_unfrozen(name, configs) -> None:
    foundation = configs[name]["model"]["foundation"]
    assert foundation["freeze"] is False
    assert foundation["frozen_modules_eval"] is False
    assert foundation["trainable_modules"] == []
    assert foundation["strict_load"] is True


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_adapter_insertion_points_match_latent_bfn(name, configs) -> None:
    config = configs[name]
    adapters = config["model"]["foundation"]["feature_adapters"]
    # decoder_level1 is impossible: foundation.py:451-462 restricts latent
    # adapters to {decoder_level3, decoder_level2}.
    assert set(adapters) == {
        "encoder_level3",
        "decoder_level3",
        "decoder_level2",
    }
    assert adapters["encoder_level3"]["type"] == "paper_spectral_gate"
    assert adapters["encoder_level3"]["channels"] == 192
    assert adapters["encoder_level3"]["initial_gate"] == 0.999
    latent = config["model"]["latent_bfn"]
    assert latent["enabled"] is True
    assert latent["feature_channels"] == 192
    for module_name, channels in (
        ("decoder_level3", 192),
        ("decoder_level2", 96),
    ):
        adapter = adapters[module_name]
        assert adapter["type"] == "latent_film_spectral"
        assert adapter["channels"] == channels
        assert adapter["trainable"] is True
        for key in ("global_dim", "spatial_dim", "hidden_channels"):
            assert adapter[key] == latent[key]
    assert set(latent["category_patterns"]) == {
        "snow",
        "outdoor_rain",
        "raindrop",
    }
    assert len(latent["category_patterns"]) == latent["num_categories"]


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_backbone_learning_rate_is_lower_than_new_modules(name, configs) -> None:
    training = configs[name]["training"]
    parameter_lrs = training["parameter_lrs"]
    assert "backbone.model" in parameter_lrs
    assert parameter_lrs["backbone.model"] < training["learning_rate"]
    # The frozen-main per-decoder groups must not survive into v26.
    assert set(parameter_lrs) == {"backbone.model"}


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_budget_lock_is_disabled(name, configs) -> None:
    assert configs[name]["training"]["require_budget_lock"] is False


@pytest.mark.parametrize("name", BALANCED_CONFIG_NAMES)
def test_sampling_is_category_balanced_and_normalized(name, configs) -> None:
    sampling = configs[name]["training"]["sampling"]
    assert sampling["mode"] == "category_balanced"
    assert sampling == EXPECTED_SAMPLING
    fractions = sampling["target_fractions"]
    assert set(fractions) == {"snow", "outdoor_rain", "raindrop"}
    assert sum(fractions.values()) == pytest.approx(1.0)
    # expected_counts is deliberately absent: samplers.py only asserts the
    # per-category equality when it is present, and v26 trains on the 95%
    # split whose counts change whenever the manifest is regenerated.
    assert "expected_counts" not in sampling


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_stage_epochs_sum_to_training_epochs(name, configs) -> None:
    training = configs[name]["training"]
    stages = training["stages"]
    assert sum(int(stage["epochs"]) for stage in stages) == int(
        training["epochs"]
    )
    # resolve_training_stages raises when the sums disagree, and the latent
    # path requires the joint stage (model.py:1254-1258).
    resolved = resolve_training_stages(training)
    assert [stage.name for stage in resolved] == ["joint"]
    assert resolved[-1].end_epoch == int(training["epochs"])


def test_main_line_keeps_four_step_bayesian_bfn(configs) -> None:
    config = configs["ssr_bfn_fullft_v26"]
    latent = config["model"]["latent_bfn"]
    core = config["model"]["core"]
    assert latent["steps"] == 4
    assert latent["update_mode"] == "bayesian"
    assert latent["accuracy_schedule"] == "linear_beta"
    assert latent["loss_mode"] == "final_mean"
    assert latent["loss_weighting"] == "adaptive_accuracy"
    assert latent["adapter_precision_mode"] == "belief"
    assert latent["trainable"] is True
    assert latent["spatial_downsample"] == 4
    assert core == {
        "require_spectral_gate": True,
        "require_iterative_bfn": True,
        "require_entropy_mask": False,
        # model.py:363-368 compares core.bfn_steps with latent_bfn.steps.
        "bfn_steps": 4,
    }
    loss = config["loss"]
    assert loss["structural_weight"] == 1.0
    assert loss["structural_loss_type"] == "mse"
    assert loss["frequency_weight"] == pytest.approx(0.05)
    assert loss["bfn_weight"] == pytest.approx(0.01)
    assert loss["latent_category_weight"] == pytest.approx(0.01)
    for key in (
        "final_weight",
        "correlation_weight",
        "adapter_weight",
        "uncertainty_weight",
    ):
        assert loss[key] == 0
    training = config["training"]
    assert training["output_dir"] == "runs/fullft_v26/main"
    assert training["epochs"] == 30
    assert training["batch_size"] == 16
    assert training["micro_batch_size"] == 4
    assert training["learning_rate"] == pytest.approx(2e-4)
    assert training["parameter_lrs"]["backbone.model"] == pytest.approx(1.5e-5)
    assert training["warmup_epochs"] == 1
    assert training["minimum_learning_rate"] == pytest.approx(1e-6)
    assert training["minimum_learning_rate_ratio"] == pytest.approx(0.1)
    assert training["schedule_minimum_mode"] == "per_group"
    assert training["weight_decay"] == pytest.approx(1e-4)
    assert training["gradient_clip"] == pytest.approx(1.0)
    assert training["betas"] == [0.9, 0.999]
    assert training["ema_decay"] == pytest.approx(0.999)
    assert training["ema_warmup_updates"] == 2000
    assert training["amp"] is True
    assert training["amp_dtype"] == "bfloat16"
    assert training["allow_tf32"] is True
    assert training["cudnn_benchmark"] is True
    assert training["reproducible_data"] is True
    assert training["validate_interval"] == 1
    assert training["save_interval"] == 1
    assert training["log_interval"] == 250
    data = config["data"]
    assert data["workers"] == 12
    assert data["train"]["patch_size"] == 256
    assert data["train"]["augment"] is True
    assert data["train"]["strict"] is True
    assert data["train"]["pairs_file"].endswith("train_pairs.tsv")
    assert data["val"]["pairs_file"].endswith("val_pairs.tsv")
    # The 903 dev pairs must not appear in the training stream: with ~11.3M
    # trainable parameters the frozen-main `pairs_file: null` full-directory
    # scan would make the dev curve track memorisation instead of
    # generalisation.
    assert data["train"]["pairs_file"] != data["val"]["pairs_file"]
    validation = config["validation"]
    assert validation["lpips"] is False
    assert validation["crop_border"] == 0
    assert validation["test_y_channel"] is True
    # The unused RGB receiver stays disabled so it cannot join the optimizer.
    assert config["model"]["bfn"]["inference_mode"] == "none"


def test_pilot_changes_only_length_and_logging(configs) -> None:
    main = dict(configs["ssr_bfn_fullft_v26"])
    pilot = dict(configs["ssr_bfn_fullft_v26_pilot"])
    assert pilot["training"]["output_dir"] == "runs/fullft_v26/pilot"
    assert pilot["training"]["epochs"] == 2
    assert pilot["training"]["stages"] == [{"name": "joint", "epochs": 2}]
    assert pilot["training"]["log_interval"] == 50
    assert pilot["model"] == main["model"]
    assert pilot["data"] == main["data"]
    assert pilot["loss"] == main["loss"]
    assert pilot["validation"] == main["validation"]
    normalized_pilot = {
        key: value
        for key, value in pilot["training"].items()
        if key not in {"output_dir", "epochs", "stages", "log_interval"}
    }
    normalized_main = {
        key: value
        for key, value in main["training"].items()
        if key not in {"output_dir", "epochs", "stages", "log_interval"}
    }
    assert normalized_pilot == normalized_main


def pixel_share(loss: dict) -> float:
    """Pixel-term share of the total objective at the pilot's raw components.

    Mirrors 【代码事实】``sgbformer/losses.py:174-183``: the total is the sum of
    ``weight * raw_component`` over the active terms, while the diagnostics dict
    (``losses.py:184-194``) that produced the raw constants is unweighted.
    """

    weighted = {
        key: float(loss[f"{key}_weight"]) * value
        for key, value in PILOT_RAW_COMPONENTS.items()
    }
    return weighted["structural"] / sum(weighted.values())


def test_pilot_weights_put_less_than_half_the_objective_on_pixels(configs) -> None:
    # The defect that v26b fixes: 46.1% pixel share, and the reconstruction of
    # the logged loss=0.00201 proves the decomposition is right.
    loss = configs["ssr_bfn_fullft_v26"]["loss"]
    total = sum(
        float(loss[f"{key}_weight"]) * value
        for key, value in PILOT_RAW_COMPONENTS.items()
    )
    assert total == pytest.approx(0.0020172, abs=1e-7)
    # The log prints 5 decimals for every term, so the reconstruction can only
    # be checked to the accumulated display rounding: struct contributes
    # +/-5e-6, freq +/-2.5e-7, bfn +/-5e-8, cls +/-5e-8 and `loss` itself
    # +/-5e-6, i.e. about +/-1.05e-5. Observed gap: 0.0020172 - 0.00201 = 7.2e-6.
    assert total == pytest.approx(0.00201, abs=1.05e-5)  # logged value
    assert pixel_share(loss) == pytest.approx(0.4610, abs=5e-4)


def test_v26b_changes_only_loss_weights_and_micro_batch(configs) -> None:
    main = configs["ssr_bfn_fullft_v26"]
    v26b = configs["ssr_bfn_fullft_v26b"]
    loss = v26b["loss"]
    assert loss["frequency_weight"] == pytest.approx(0.01)
    assert loss["bfn_weight"] == pytest.approx(0.004)
    assert loss["latent_category_weight"] == pytest.approx(0.002)
    assert loss["structural_weight"] == pytest.approx(1.0)
    assert loss["structural_loss_type"] == "mse"
    training = v26b["training"]
    assert training["micro_batch_size"] == 2
    assert training["output_dir"] == "runs/fullft_v26b/main"
    assert training["epochs"] == 30

    # Nothing else moved: same model, same data, same validation, and the only
    # training keys that differ are output_dir and micro_batch_size. The global
    # batch is unchanged, so train.py:421-430 only changes accumulation 4 -> 8.
    assert v26b["model"] == main["model"]
    assert v26b["data"] == main["data"]
    assert v26b["validation"] == main["validation"]
    assert training["batch_size"] == main["training"]["batch_size"] == 16
    assert training["learning_rate"] == pytest.approx(2e-4)
    assert training["parameter_lrs"]["backbone.model"] == pytest.approx(1.5e-5)
    assert training["warmup_epochs"] == 1
    assert training["minimum_learning_rate_ratio"] == pytest.approx(0.1)
    differing_training = {
        key
        for key in set(training) | set(main["training"])
        if training.get(key) != main["training"].get(key)
    }
    assert differing_training == {"output_dir", "micro_batch_size"}
    differing_loss = {
        key
        for key in set(loss) | set(main["loss"])
        if loss.get(key) != main["loss"].get(key)
    }
    assert differing_loss == {
        "frequency_weight",
        "bfn_weight",
        "latent_category_weight",
    }


def test_v26b_inherits_the_full_unfreeze_and_balanced_sampling(configs) -> None:
    v26b = configs["ssr_bfn_fullft_v26b"]
    assert v26b["model"]["foundation"]["freeze"] is False
    assert v26b["model"]["foundation"]["trainable_modules"] == []
    assert v26b["model"]["foundation"]["frozen_modules_eval"] is False
    assert v26b["training"]["sampling"]["mode"] == "category_balanced"
    assert v26b["training"]["sampling"]["target_fractions"] == {
        "snow": 0.50,
        "outdoor_rain": 0.25,
        "raindrop": 0.25,
    }
    assert v26b["data"]["train"]["pairs_file"].endswith("train_pairs.tsv")
    assert (
        v26b["data"]["train"]["pairs_file"] != v26b["data"]["val"]["pairs_file"]
    )
    assert v26b["model"]["latent_bfn"]["steps"] == 4
    assert v26b["model"]["latent_bfn"]["update_mode"] == "bayesian"


def test_v26b_pixel_share_is_frozen_between_74_and_77_percent(configs) -> None:
    # Guard rail: this is the whole point of v26b. freq 0.01x0.00944 = 9.44e-5,
    # bfn 0.004x0.04451 = 1.7804e-4, cls 0.002x0.01701 = 3.402e-5
    # -> 3.0646e-4 auxiliary vs 9.300e-4 pixel (32.95%), total 1.23646e-3,
    # pixel share 0.75214. Anyone re-tuning the weights must justify it here.
    share = pixel_share(configs["ssr_bfn_fullft_v26b"]["loss"])
    assert 0.74 < share < 0.77
    assert share == pytest.approx(0.75214, abs=5e-4)
    assert share > pixel_share(configs["ssr_bfn_fullft_v26"]["loss"])


def test_v26b_proxy_is_six_epochs_of_the_same_recipe(configs) -> None:
    proxy = configs["ssr_bfn_fullft_v26b_proxy"]
    v26b = configs["ssr_bfn_fullft_v26b"]
    assert proxy["training"]["output_dir"] == "runs/fullft_v26b/proxy"
    assert proxy["training"]["epochs"] == 6
    assert proxy["training"]["stages"] == [{"name": "joint", "epochs": 6}]
    assert sum(int(stage["epochs"]) for stage in proxy["training"]["stages"]) == 6
    # Three-level _base_ chain (proxy -> v26b -> v26 -> ../ssr_bfn_5090.yaml)
    # resolved by sgbformer/config.py:36-46.
    assert proxy["training"]["micro_batch_size"] == 2
    assert proxy["loss"] == v26b["loss"]
    assert proxy["model"] == v26b["model"]
    assert proxy["data"] == v26b["data"]
    assert proxy["training"]["save_interval"] == 1  # every epoch_000N.pt exists
    differing = {
        key
        for key in set(proxy["training"]) | set(v26b["training"])
        if proxy["training"].get(key) != v26b["training"].get(key)
    }
    assert differing == {"output_dir", "epochs", "stages"}


def test_v26c_nobalance_proxy_changes_only_the_sampler_mode(configs) -> None:
    balanced = configs["ssr_bfn_fullft_v26b_proxy"]
    natural = configs["ssr_bfn_fullft_v26c_nobalance_proxy"]

    assert natural["training"]["output_dir"] == (
        "runs/fullft_v26c/nobalance_proxy"
    )
    assert natural["training"]["sampling"]["mode"] == "shuffle"
    assert balanced["training"]["sampling"]["mode"] == "category_balanced"

    # Recursive config merging leaves the balanced-only keys present, but
    # train.py dispatches on mode and its shuffle branch never reads them.
    assert natural["training"]["sampling"]["category_patterns"] == (
        balanced["training"]["sampling"]["category_patterns"]
    )
    assert natural["training"]["sampling"]["target_fractions"] == (
        balanced["training"]["sampling"]["target_fractions"]
    )

    # Preserve the category objective. Removing it together with the sampler
    # would change two variables and make the proxy non-causal.
    assert natural["loss"] == balanced["loss"]
    assert natural["loss"]["latent_category_weight"] == pytest.approx(0.002)
    assert natural["model"] == balanced["model"]
    assert natural["data"] == balanced["data"]
    assert natural["validation"] == balanced["validation"]
    assert natural["training"]["epochs"] == balanced["training"]["epochs"] == 6
    assert natural["training"]["stages"] == balanced["training"]["stages"]
    assert natural["training"]["micro_batch_size"] == 2
    assert natural["training"]["batch_size"] == 16
    assert natural["training"]["parameter_lrs"] == (
        balanced["training"]["parameter_lrs"]
    )
    assert natural["training"]["reproducible_data"] is True

    differing_training = {
        key
        for key in set(natural["training"]) | set(balanced["training"])
        if natural["training"].get(key) != balanced["training"].get(key)
    }
    assert differing_training == {"output_dir", "sampling"}
    differing_sampling = {
        key
        for key in (
            set(natural["training"]["sampling"])
            | set(balanced["training"]["sampling"])
        )
        if natural["training"]["sampling"].get(key)
        != balanced["training"]["sampling"].get(key)
    }
    assert differing_sampling == {"mode"}


def test_v26d_lrlow_proxy_changes_only_the_backbone_lr(configs) -> None:
    reference = configs["ssr_bfn_fullft_v26b_proxy"]
    low_lr = configs["ssr_bfn_fullft_v26d_lrlow_proxy"]

    assert low_lr["training"]["output_dir"] == "runs/fullft_v26d/lrlow_proxy"
    assert low_lr["training"]["parameter_lrs"] == {
        "backbone.model": pytest.approx(5e-6),
    }
    assert reference["training"]["parameter_lrs"] == {
        "backbone.model": pytest.approx(1.5e-5),
    }

    assert low_lr["loss"] == reference["loss"]
    assert low_lr["model"] == reference["model"]
    assert low_lr["data"] == reference["data"]
    assert low_lr["validation"] == reference["validation"]
    assert low_lr["training"]["sampling"] == reference["training"]["sampling"]
    assert low_lr["training"]["epochs"] == reference["training"]["epochs"] == 6
    assert low_lr["training"]["stages"] == reference["training"]["stages"]
    assert low_lr["training"]["micro_batch_size"] == 2
    assert low_lr["training"]["batch_size"] == 16
    assert low_lr["training"]["learning_rate"] == pytest.approx(2e-4)

    differing_training = {
        key
        for key in set(low_lr["training"]) | set(reference["training"])
        if low_lr["training"].get(key) != reference["training"].get(key)
    }
    assert differing_training == {"output_dir", "parameter_lrs"}
    differing_parameter_lrs = {
        key
        for key in (
            set(low_lr["training"]["parameter_lrs"])
            | set(reference["training"]["parameter_lrs"])
        )
        if low_lr["training"]["parameter_lrs"].get(key)
        != reference["training"]["parameter_lrs"].get(key)
    }
    assert differing_parameter_lrs == {"backbone.model"}

    final_factor = cosine_warmup_factor(
        5,
        total_epochs=6,
        warmup_epochs=1,
        minimum_ratio=0.1,
    )
    assert 5e-6 * final_factor == pytest.approx(9.2971e-7, abs=1e-11)


def test_proxy_anneals_while_the_pilot_never_did() -> None:
    """【代码事实】sgbformer/training.py:87-99, evaluated for each run length."""

    def factor(epoch: int, total_epochs: int) -> float:
        return cosine_warmup_factor(
            epoch,
            total_epochs=total_epochs,
            warmup_epochs=1,
            minimum_ratio=0.1,
        )

    # 2-epoch pilot: both epochs at 100% of peak LR, zero annealing.
    assert factor(0, 2) == pytest.approx(1.0)
    assert factor(1, 2) == pytest.approx(1.0)
    # 6-epoch proxy: real decay down to 18.6% (backbone LR 2.789e-6).
    assert factor(5, 6) == pytest.approx(0.185942, abs=1e-6)
    assert 1.5e-5 * factor(5, 6) == pytest.approx(2.7891e-6, abs=1e-10)
    # 30-epoch main line: decays to 10.264% (backbone LR 1.5396e-6).
    assert factor(29, 30) == pytest.approx(0.102638, abs=1e-6)
    assert 1.5e-5 * factor(29, 30) == pytest.approx(1.5396e-6, abs=1e-10)
