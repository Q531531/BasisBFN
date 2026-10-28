from copy import deepcopy

import pytest
import torch
import yaml

from sgbformer.checkpoint import save_checkpoint
from sgbformer.inference import (
    load_model_for_inference,
    restoration_tile_count,
    restore_structural_tensor,
    restore_tensor,
    validate_inference_config,
)
from sgbformer.models import SGBFormer


def model_config():
    return {
        "model": {
            "semantic": {"encoder": "none", "embedding_dim": 1},
            "foundation": {
                "type": "external_ssr", "repo_path": "/old/SSR", "checkpoint": "/old/a.pth",
                "feature_adapters": {
                    "decoder_level1": {"risk_log_ratio_mode": "smooth"}
                },
            },
        },
    }


def test_inference_configuration_only_allows_path_relocation():
    saved = model_config()
    relocated = deepcopy(saved)
    relocated["model"]["foundation"].update(repo_path="/new/SSR", checkpoint="/new/a.pth")
    relocated["training"] = {"output_dir": "/unused"}
    validate_inference_config(saved, relocated)
    for key, value in (("risk_log_ratio_mode", "clamp"), ("spectral_reference_size", 256)):
        changed = deepcopy(relocated)
        changed["model"]["foundation"]["feature_adapters"]["decoder_level1"][key] = value
        with pytest.raises(ValueError, match="changes the checkpoint model"):
            validate_inference_config(saved, changed)


def test_actual_inference_loader_rejects_semantic_override(tmp_path):
    config = {
        "model": {
            "dim": 8, "num_blocks": [1, 1, 1, 1], "refinement_blocks": 1,
            "heads": [1, 2, 4, 8],
            "semantic": {"encoder": "none", "embedding_dim": 16},
            "bfn": {"inference_mode": "none", "mask_mode": "none"},
        },
    }
    model = SGBFormer(**config["model"])
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model, config=config, epoch=0, global_step=0)
    override = deepcopy(config)
    override["model"]["bfn"]["mask_mode"] = "entropy"
    path = tmp_path / "override.yaml"
    path.write_text(yaml.safe_dump(override))
    with pytest.raises(ValueError, match="changes the checkpoint model"):
        load_model_for_inference(checkpoint, torch.device("cpu"), config_path=path)
    path.write_text(yaml.safe_dump(config))
    restored, _ = load_model_for_inference(checkpoint, torch.device("cpu"), config_path=path)
    assert restored.refinement_mode == "none"


class IdentityRestorer:
    def encode_semantic(self, image):
        return image.new_zeros((image.shape[0], 1))

    def restore(self, image, **kwargs):
        return image

    def restore_structural(self, image, **kwargs):
        return image


def test_tiled_identity_has_no_uncovered_pixels():
    for shape in ((1, 3, 17, 31), (1, 3, 3, 27), (1, 3, 2, 2)):
        image = torch.rand(shape)
        result = restore_tensor(IdentityRestorer(), image, tile=8, overlap=3)
        torch.testing.assert_close(result, image, rtol=1e-6, atol=1e-7)
        structural = restore_structural_tensor(
            IdentityRestorer(),
            image,
            tile=8,
            overlap=3,
        )
        torch.testing.assert_close(structural, image, rtol=1e-6, atol=1e-7)


def test_restoration_tile_count_matches_grid() -> None:
    assert restoration_tile_count(256, 256, None, 32) == 1
    assert restoration_tile_count(256, 256, 256, 32) == 1
    assert restoration_tile_count(512, 512, 256, 32) == 9
    assert restoration_tile_count(300, 700, 256, 32) == 6


def test_restore_tensor_forwards_bfn_gain() -> None:
    class GainRestorer(IdentityRestorer):
        def restore(self, image, residual_gain=1.0, **kwargs):
            return image * residual_gain

    image = torch.ones(1, 3, 8, 8)
    result = restore_tensor(
        GainRestorer(),
        image,
        residual_gain=0.25,
    )
    torch.testing.assert_close(result, torch.full_like(result, 0.25))


@pytest.mark.parametrize("tile,overlap", [(0, 0), (-8, 0), (8, -1), (8, 8)])
def test_tile_parameters_rejected_even_on_small_images(tile, overlap):
    with pytest.raises(ValueError, match="overlap"):
        restore_tensor(IdentityRestorer(), torch.rand(1, 3, 2, 2), tile=tile, overlap=overlap)
