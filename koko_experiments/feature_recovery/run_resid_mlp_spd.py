from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fire
import torch

from spd.configs import Config
from spd.experiments.resid_mlp.models import ResidMLP
from spd.run_spd import run_experiment
from spd.settings import SPD_OUT_DIR
from spd.utils.data_utils import DatasetGeneratedDataLoader
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import set_seed
from spd.utils.run_utils import generate_run_id, save_file

from koko_experiments.feature_recovery.feature_datasets import DictionaryFeatureDataset
from koko_experiments.feature_recovery.results import load_feature_recovery_target_bundle
from koko_experiments.feature_recovery.synthetic import ActivationGenerator, FeatureDictionary


def _prepare_config(spd_config_path: Path, target_run_dir: Path) -> Config:
    config = Config.from_file(spd_config_path)
    config_dict = config.model_dump(mode="json")
    config_dict["pretrained_model_class"] = "spd.experiments.resid_mlp.models.ResidMLP"
    config_dict["pretrained_model_path"] = str(target_run_dir / "resid_mlp_feature_recovery.pth")
    return Config.model_validate(config_dict)


def main(spd_config_path: str, target_run_dir: str, run_id: str | None = None) -> None:
    target_dir = Path(target_run_dir)
    train_config, state_dict, feature_vectors, label_coeffs = load_feature_recovery_target_bundle(target_dir)
    spd_config = _prepare_config(Path(spd_config_path), target_dir)

    set_seed(spd_config.seed)
    device = get_device()

    target_model = ResidMLP(train_config.resid_mlp_model_config)
    target_model.load_state_dict(state_dict)
    target_model.to(device)
    target_model.eval()

    feature_dict = FeatureDictionary(feature_vectors)
    activation_generator = ActivationGenerator(train_config.dataset.generator, feature_dict.num_features)
    dataset = DictionaryFeatureDataset(
        feature_dict=feature_dict,
        activation_generator=activation_generator,
        label_type=train_config.label_type,
        label_coeffs=label_coeffs,
        act_fn_name=train_config.resid_mlp_model_config.act_fn_name,
        device=device,
    )
    train_loader = DatasetGeneratedDataLoader(dataset, batch_size=spd_config.batch_size, shuffle=False)
    eval_loader = DatasetGeneratedDataLoader(dataset, batch_size=spd_config.eval_batch_size, shuffle=False)

    run_id = run_id or generate_run_id("spd")
    run_experiment(
        target_model=target_model,
        config=spd_config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        experiment_tag="feature_recovery_resid_mlp",
        run_id=run_id,
        launch_id=None,
        evals_id=None,
        sweep_params=None,
        target_model_train_config=train_config,
    )
    run_dir = SPD_OUT_DIR / "spd" / run_id
    save_file({"feature_vectors": feature_vectors}, run_dir / "feature_dictionary.pt")
    save_file(label_coeffs.detach().cpu().tolist(), run_dir / "label_coeffs.json")
    print(run_dir)


if __name__ == "__main__":
    fire.Fire(main)
