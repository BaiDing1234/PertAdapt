from pathlib import Path
from typing import Any, Literal

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
import lightning as L

from modelgenerator.structure_tokenizer.models.esmfold_decoder import (
    ESMFoldDecoderLightning,
)
from modelgenerator.structure_tokenizer.utils.types import PathLike
from modelgenerator.structure_tokenizer.datasets.protein import Protein

from modelgenerator.structure_tokenizer.configs.data_configs import (
    ProteinDataConfig,
    StructTokensDataConfig,
)
from modelgenerator.structure_tokenizer.models.structure_tokenizer_lightning import (
    StructureTokenizerLightning,
)
from modelgenerator.structure_tokenizer.utils.geometry.kabsch import (
    find_rigid_alignment,
)


def _align(
    atom_positions: torch.Tensor,
    ref_atom_positions: torch.Tensor,
    atom_mask: torch.Tensor,
) -> torch.Tensor:
    backbone_atom_mask = atom_mask.clone()
    backbone_atom_mask[:, 3:] = False
    masked_atom_positions = atom_positions[atom_mask]
    masked_ref_atom_positions = ref_atom_positions[atom_mask]
    rotation, trans = find_rigid_alignment(
        A=masked_atom_positions, B=masked_ref_atom_positions
    )
    aligned_atom_positions = atom_positions @ rotation.T + trans
    return aligned_atom_positions


def _extract_protein_entity_chain_ids_from_name(
    name: str,
) -> tuple[str, str | None, str]:
    ids = name.split("_")
    match len(ids):
        case 2:  # no entity id
            return (ids[0], None, ids[1]) if ids[1] != "nan" else (ids[0], None, "")
        case 3:  # entity id
            return ids[0], ids[1], ids[2]
        case _:
            raise ValueError(f"Invalid protein entity chain name: {name}")


class WriterPDBCallback(L.Callback):
    def __init__(self, dirpath: PathLike) -> None:
        super().__init__()
        self.dirpath = Path(dirpath)
        self.dirpath_predict: dict[str, Path] = dict()
        self.predict_id_names = None
        self.mode: Literal["structure_tokenizer", "structure_decoder"] | None = None

    def setup(
        self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str
    ) -> None:
        assert isinstance(pl_module, ESMFoldDecoderLightning) or isinstance(
            pl_module, StructureTokenizerLightning
        ), "pl_module must be ESMFoldDecoderLightning or StructureTokenizerLightning"
        config = trainer.datamodule.config
        self.predict_id_names = config.dataloader_idx_to_name
        if isinstance(config, ProteinDataConfig):
            self.mode = "structure_tokenizer"
        elif isinstance(config, StructTokensDataConfig):
            self.mode = "structure_decoder"
        else:
            raise ValueError(f"Unsupported config type: {type(config)}")

    def on_predict_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if self.predict_id_names:
            for name in self.predict_id_names.values():
                dirpath_predict_dataset = Path(self.dirpath) / f"{name}_pdb_files"
                dirpath_predict_dataset.mkdir(exist_ok=True, parents=True)
                self.dirpath_predict[name] = dirpath_predict_dataset

    def on_predict_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        name = self.predict_id_names[dataloader_idx]
        match self.mode:
            case "structure_tokenizer":
                self._write_pdb_from_structure_tokenizer(
                    batch=batch,
                    outputs=outputs,
                    dirpath=self.dirpath_predict[name],
                )
            case "structure_decoder":
                self._write_pdb_from_esmfold_decoder(
                    batch=batch,
                    outputs=outputs,
                    dirpath=self.dirpath_predict[name],
                )
            case _:
                raise ValueError(f"Unsupported mode: {self.mode}")

    @staticmethod
    def _write_pdb_from_esmfold_decoder(
        batch: dict[str, list[str] | torch.Tensor],
        outputs: STEP_OUTPUT,
        dirpath: PathLike,
    ) -> None:
        dirpath = Path(dirpath)
        # Extract items
        batch_names = batch["name"]
        attention_mask = batch["attention_mask"]
        batch_residue_index = batch["residue_index"]
        batch_atom_masks = outputs["atom37_atom_exists"]
        batch_atom_positions = outputs["coords"]
        batch_aatype = outputs["aatype"]
        seqlen = attention_mask.type(torch.int64).sum(dim=-1)

        with torch.no_grad():
            for (
                name,
                atom_positions,
                atom_masks,
                aatype,
                residue_index,
                l,
            ) in zip(
                batch_names,
                batch_atom_positions,
                batch_atom_masks,
                batch_aatype,
                batch_residue_index,
                seqlen,
            ):
                protein_id, entity_id, chain_id = (
                    _extract_protein_entity_chain_ids_from_name(name=name)
                )
                protein = Protein(
                    id=protein_id,
                    chain_id=chain_id,
                    entity_id=entity_id,
                    atom_positions=atom_positions[:l, :, :].detach().cpu().numpy(),
                    atom_mask=atom_masks[:l, :].detach().cpu().numpy(),
                    aatype=aatype[:l].detach().cpu().numpy(),
                    residue_index=residue_index[:l].detach().cpu().numpy(),
                    b_factors=None,
                    plddt=None,
                    resolution=0,
                )
                protein_str_id = str(protein)
                protein.to_pdb(path=dirpath / f"{protein_str_id}_output.pdb")

    @staticmethod
    def _write_pdb_from_structure_tokenizer(
        batch: dict[str, list[str] | torch.Tensor],
        outputs: STEP_OUTPUT,
        dirpath: PathLike,
    ) -> None:
        dirpath = Path(dirpath)
        # Extract items
        batch_protein_ids = batch["id"]
        batch_entity_ids = batch["entity_id"]
        batch_chain_ids = batch["chain_id"]
        batch_true_atom_positions = batch["atom_positions"]
        batch_atom_positions = outputs["coords"]
        batch_atom_masks = batch["atom_masks"]
        batch_aatype = batch["aatype"]
        batch_residue_index = batch["residue_index"]
        attention_mask = batch["attention_mask"]
        seqlen = attention_mask.type(torch.int64).sum(dim=-1)
        #
        input_proteins_paths = list()
        output_proteins_paths = list()

        with torch.no_grad():
            for (
                protein_id,
                entity_id,
                chain_id,
                true_atom_positions,
                atom_positions,
                atom_masks,
                aatype,
                residue_index,
                l,
            ) in zip(
                batch_protein_ids,
                batch_entity_ids,
                batch_chain_ids,
                batch_true_atom_positions,
                batch_atom_positions,
                batch_atom_masks,
                batch_aatype,
                batch_residue_index,
                seqlen,
            ):
                aligned_atom_positions = _align(
                    atom_positions=atom_positions,
                    ref_atom_positions=true_atom_positions,
                    atom_mask=atom_masks,
                )
                aligned_protein = Protein(
                    id=protein_id,
                    chain_id=chain_id,
                    entity_id=entity_id,
                    atom_positions=aligned_atom_positions[:l, :, :]
                    .detach()
                    .cpu()
                    .numpy(),
                    atom_mask=atom_masks[:l, :].detach().cpu().numpy(),
                    aatype=aatype[:l].detach().cpu().numpy(),
                    residue_index=residue_index[:l].detach().cpu().numpy(),
                    b_factors=None,
                    plddt=None,
                    resolution=0,
                )
                ref_protein = Protein(
                    id=protein_id,
                    chain_id=chain_id,
                    entity_id=entity_id,
                    atom_positions=true_atom_positions[:l, :, :].detach().cpu().numpy(),
                    atom_mask=atom_masks[:l, :].detach().cpu().numpy(),
                    aatype=aatype[:l].detach().cpu().numpy(),
                    residue_index=residue_index[:l].detach().cpu().numpy(),
                    b_factors=None,
                    plddt=None,
                    resolution=0,
                )
                protein_str_id = str(ref_protein)
                ref_protein.to_pdb(path=dirpath / f"{protein_str_id}_input.pdb")
                aligned_protein.to_pdb(path=dirpath / f"{protein_str_id}_output.pdb")
                input_proteins_paths.append(dirpath / f"{protein_str_id}_input.pdb")
                output_proteins_paths.append(dirpath / f"{protein_str_id}_output.pdb")
