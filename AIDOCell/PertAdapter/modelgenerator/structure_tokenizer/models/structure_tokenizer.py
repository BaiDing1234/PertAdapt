from dataclasses import asdict

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from modelgenerator.structure_tokenizer.configs.models_configs import (
    StructureTokenizerConfig,
)
from modelgenerator.structure_tokenizer.models.equiformer_encoder import (
    EquiformerEncoder,
)
from modelgenerator.structure_tokenizer.models.esmfold_decoder import ESMFoldDecoder
from modelgenerator.structure_tokenizer.utils.init_params import (
    init_params_recursively_,
)
from modelgenerator.structure_tokenizer.utils.misc import get_config_from_dict


class StructureTokenizerModel(
    nn.Module,
    PyTorchModelHubMixin,
    # library_name="",
    # repo_url="",
    # docs_url="",
    # license="",
    tags=["biology", "genbio", "protein"],
):
    def __init__(self, config: StructureTokenizerConfig):
        super().__init__()
        config = get_config_from_dict(
            config_dict=asdict(config), config=StructureTokenizerConfig
        )
        self.encoder = EquiformerEncoder(config.encoder_config)
        self.decoder = ESMFoldDecoder(config.decoder_config)
        self.frozen_codebook = True

    def reset_parameters(self):
        init_params_recursively_(self)

    @property
    def frozen_codebook(self) -> bool:
        return self._frozen_codebook

    @frozen_codebook.setter
    def frozen_codebook(self, frozen: bool) -> None:
        self.encoder.quantize.frozen = frozen
        self._frozen_codebook = frozen

    def forward(
        self,
        atom_positions: torch.Tensor,
        aatype: torch.Tensor,
        attention_mask: torch.Tensor,
        residue_index: torch.Tensor,
        num_recycles: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        encoder output
        {
            "emb": emb,
            "raw_qx": raw_qx,
            "commitment_loss": commitment_loss,
            "logits": logits,
            "idx": idx,
            "prob": prob,
            "ppl": ppl,
            "dead_codes": nb dead codes,
        }

        decoder output
             {'frames': torch.Size([8, 2, 500, 7]),
             'sidechain_frames': torch.Size([8, 2, 500, 8, 4, 4]),
             'unnormalized_angles': torch.Size([8, 2, 500, 7, 2]),
             'angles': torch.Size([8, 2, 500, 7, 2]),
             'positions': torch.Size([8, 2, 500, 14, 3]),
             'states': torch.Size([8, 2, 500, 384]),
             's_s': torch.Size([2, 500, 768]),
             's_z': torch.Size([2, 500, 500, 128]),
             'distogram_logits': torch.Size([2, 500, 500, 64]),
             'lm_logits': torch.Size([2, 500, 24]),
             'aatype': torch.Size([2, 500]),
             'atom14_atom_exists': torch.Size([2, 500, 14]),
             'residx_atom14_to_atom37': torch.Size([2, 500, 14]),
             'residx_atom37_to_atom14': torch.Size([2, 500, 37]),
             'atom37_atom_exists': torch.Size([2, 500, 37]),
             'residue_index': torch.Size([2, 500]),
             'lddt_head': torch.Size([8, 2, 500, 37, 50]),
             'plddt': torch.Size([2, 500, 37]),
             'ptm_logits': torch.Size([2, 500, 500, 64]),
             'ptm': torch.Size([2]),
             'aligned_confidence_probs': torch.Size([2, 500, 500, 64]),
             'predicted_aligned_error': torch.Size([2, 500, 500]),
             'max_predicted_aligned_error': torch.Size([]),
             'coords': torch.Size([2, 500, 37, 3]),
             "qx": qx,
             }

        """
        output = dict()
        encoder_output = self.encoder(
            atom_positions=atom_positions,
            attention_mask=attention_mask,
            residue_index=residue_index,
        )
        output.update(encoder_output)
        decoder_output = self.decoder(
            s_s_0=encoder_output["raw_qx"],
            aatype=aatype,
            attention_mask=attention_mask,
            residue_index=residue_index,
            num_recycles=num_recycles,
        )
        output.update(decoder_output)
        return output

    @property
    def codebook(self) -> torch.Tensor:
        return self.encoder.codebook

    def freeze_unused_params(self) -> list[str]:
        enc_frozen = self.encoder.freeze_unused_params()
        decoder_frozen = self.decoder.freeze_unused_params()
        return enc_frozen + decoder_frozen
