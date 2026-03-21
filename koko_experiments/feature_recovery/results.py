import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from spd.experiments.resid_mlp.models import ResidMLP
from spd.experiments.tms.models import TMSModel
from spd.utils.run_utils import save_file

from .configs import FeatureRecoveryExperimentConfig
from .synthetic import FeatureDictionary


def _serialize(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _serialize(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    return obj


def save_unified_target_bundle(
    out_dir: Path,
    config: FeatureRecoveryExperimentConfig,
    model: TMSModel | ResidMLP,
    feature_dict: FeatureDictionary,
    label_coeffs: Tensor,
    summary: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    config.to_file(out_dir / "feature_recovery_config.yaml")
    save_file({"model_type": config.target.model_type}, out_dir / "bundle_metadata.json", indent=2)
    checkpoint_name = (
        "tms_feature_recovery.pth"
        if config.target.model_type == "tms"
        else "resid_mlp_feature_recovery.pth"
    )
    save_file(model.state_dict(), out_dir / checkpoint_name)
    save_file(
        {"feature_vectors": feature_dict.feature_vectors.detach().cpu()},
        out_dir / "feature_dictionary.pt",
    )
    save_file(label_coeffs.detach().cpu().tolist(), out_dir / "label_coeffs.json")
    save_file(_serialize(summary), out_dir / "summary.json", indent=2)


def load_unified_target_bundle(
    out_dir: Path,
) -> tuple[FeatureRecoveryExperimentConfig, str, dict[str, Tensor], Tensor, Tensor]:
    config = FeatureRecoveryExperimentConfig.from_file(out_dir / "feature_recovery_config.yaml")
    model_type = json.loads((out_dir / "bundle_metadata.json").read_text())["model_type"]
    checkpoint_name = (
        "tms_feature_recovery.pth" if model_type == "tms" else "resid_mlp_feature_recovery.pth"
    )
    state_dict = torch.load(out_dir / checkpoint_name, map_location="cpu", weights_only=True)
    feature_vectors = torch.load(
        out_dir / "feature_dictionary.pt",
        map_location="cpu",
        weights_only=True,
    )["feature_vectors"]
    label_coeffs = torch.tensor(
        json.loads((out_dir / "label_coeffs.json").read_text()),
        dtype=torch.float32,
    )
    return config, model_type, state_dict, feature_vectors, label_coeffs
