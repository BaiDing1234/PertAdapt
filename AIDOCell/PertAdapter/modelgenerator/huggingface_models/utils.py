import os
import warnings
from jsonargparse import CLI
from typing import Type
from modelgenerator.huggingface_models.genbio.modeling_genbio import (
    GenBioModel,
)
from modelgenerator.tasks import TaskInterface
from lightning.pytorch.cli import instantiate_module
from lightning.pytorch.core.mixins.hparams_mixin import _given_hyperparameters_context


def instantiate_module_preserve_hparams(class_type, config):
    """Instantiate model while preserving hyperparameters.

    `load_from_checkpoint` by default uses the instantiate_module
    function that alters hyperparameters previously saved in the
    checkpoint. This function patches it to preserve the hyperparameters,
    which is required to initialize the model so that huggingface's
    `from_pretrained` works correctly.
    """
    with _given_hyperparameters_context(
        config, "lightning.pytorch.cli.instantiate_module"
    ):
        return instantiate_module(class_type, config)


def convert_to_hf_model_from_ckpt(
    task_class: Type[TaskInterface],
    ckpt_path: str | os.PathLike,
    dest_dir: str | os.PathLike,
    push_to_hub: bool = False,
    repo_id: str = None,
):
    """
    Create a huggingface model from mgen checkpoint. Optionally push to huggingface hub.

    Args:
        task_class: The task class used to load the checkpoint.
        ckpt_path: The path to the checkpoint to be loaded.
        dest_dir: The path to save the huggingface model.
        push_to_hub: Whether to push the model to huggingface hub.
    """
    instantiator_path = (
        "modelgenerator.huggingface_models.utils.instantiate_module_preserve_hparams"
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Found keys that are*")
        task = task_class.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            _instantiator=instantiator_path,
            strict_loading=False,
        )
    hf_model = GenBioModel.from_genbio_model(task)
    checkpoint = {"state_dict": hf_model.genbio_model.state_dict()}
    hf_model.genbio_model.on_save_checkpoint(checkpoint)
    state_dict = {f"genbio_model.{k}": v for k, v in checkpoint["state_dict"].items()}
    hf_model.save_pretrained(
        dest_dir, state_dict=state_dict, push_to_hub=push_to_hub, repo_id=repo_id
    )
    return hf_model


def hf_cli_main():
    cli = CLI(
        components={"convert": convert_to_hf_model_from_ckpt},
        as_positional=False,
    )


if __name__ == "__main__":
    hf_cli_main()
