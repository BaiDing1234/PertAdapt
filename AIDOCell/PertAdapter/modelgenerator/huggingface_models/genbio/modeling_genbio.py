import warnings

from lightning.pytorch import LightningModule
from lightning.pytorch.core.saving import _load_state
from transformers import PreTrainedModel, PretrainedConfig


class GenBioConfig(PretrainedConfig):
    model_type = "genbio"

    def __init__(self, hparams=None, **kwargs):
        self.hparams = hparams
        super().__init__(**kwargs)


class GenBioModel(PreTrainedModel):
    config_class = GenBioConfig

    def __init__(self, config: GenBioConfig, genbio_model=None, **kwargs):
        super().__init__(config, **kwargs)
        # if genbio_model is provided, we don't need to initialize it
        if genbio_model is not None:
            self.genbio_model = genbio_model
            return
        # otherwise, initialize the model from hyperparameters
        cls_path = config.hparams["_class_path"]
        module_path, name = cls_path.rsplit(".", 1)
        genbio_cls = getattr(__import__(module_path, fromlist=[name]), name)
        checkpoint = {
            LightningModule.CHECKPOINT_HYPER_PARAMS_KEY: config.hparams,
            "state_dict": {},
        }
        # TODO: _load_state is a private function and it throws a warning for an
        # empty state_dict. We need a fucntion to intialize our model; this
        # is the only choice we have for now.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Found keys that are*")
            self.genbio_model = _load_state(
                genbio_cls, checkpoint, strict_loading=False
            )

    @classmethod
    def from_genbio_model(cls, model: LightningModule):
        return cls(GenBioConfig(hparams=model.hparams), genbio_model=model)

    def forward(self, *args, **kwargs):
        return self.genbio_model(*args, **kwargs)


GenBioModel.register_for_auto_class("AutoModel")
GenBioConfig.register_for_auto_class("AutoConfig")
