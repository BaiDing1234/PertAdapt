import dataclasses
import gzip
import logging
import tarfile
from functools import cached_property
from pathlib import Path
from typing import Literal

import torch
from biotite.structure.io import pdbx, pdb
import biotite.structure as bs
import numpy as np

from modelgenerator.structure_tokenizer.utils.constants import residue_constants as RC
from modelgenerator.structure_tokenizer.utils.shape_utils import (
    slice_python_object_as_numpy,
)
from modelgenerator.structure_tokenizer.utils.types import PathLike

MIN_NB_RES: int = 5


logger = logging.getLogger(__name__)


def get_resolution(cif_file: pdbx.CIFFile) -> float:
    pdb_id = list(cif_file.keys())[0]
    cif_block = cif_file[pdb_id]
    resolution = 0.0
    for res_key in (
        "refine.ls_d_res_high",
        "em_3d_reconstruction.resolution",
        "reflns.d_resolution_high",
    ):
        if res_key.split(".")[0] in cif_block:
            try:
                key_1, key_2 = res_key.split(".")
                raw_resolution = cif_block[key_1][key_2].as_item()
                resolution = float(raw_resolution)
                return resolution
            except ValueError:
                pass
    return resolution


class ProteinChainEmpty(Exception):
    pass


class ProteinChainTooSmall(Exception):
    pass


def _process_atom_array(
    atom_array: bs.AtomArray,
) -> tuple[np.ndarray, ...]:
    sequence = "".join(
        (r if len(r := RC.restype_3to1.get(monomer[0].res_name, "X")) == 1 else "X")
        for monomer in bs.residue_iter(atom_array)
    )
    num_res = len(sequence)
    atom_positions = np.full(
        [num_res, RC.atom_type_num, 3],
        0,  # np.nan,
        dtype=np.float32,
    )
    atom_mask = np.full(
        [num_res, RC.atom_type_num],
        False,
        dtype=bool,
    )
    residue_index = np.full([num_res], -1, dtype=np.int64)
    confidence = np.ones(
        [num_res],
        dtype=np.float32,
    )

    for i, res in enumerate(bs.residue_iter(atom_array)):
        label_seq_id = getattr(res[0], "label_seq_id", None)
        res_index = int(label_seq_id) if label_seq_id is not None else res[0].res_id
        residue_index[i] = res_index

        # Atom level features
        for atom in res:
            atom_name = atom.atom_name
            # if atom_name == "SE" and atom.res_name == "MSE":
            #     # Put the coords of the selenium atom in the sulphur column
            #     atom_name = "SD"
            if atom_name in RC.atom_order:
                atom_positions[i, RC.atom_order[atom_name]] = atom.coord
                atom_mask[i, RC.atom_order[atom_name]] = True
                if atom_name == "CA":
                    confidence[i] = atom.b_factor

    # Remove invalid backbones
    valid_backbone_mask = atom_mask[:, :3].copy().all(axis=1)
    valid_backbone_indices = np.nonzero(valid_backbone_mask)[0]
    sequence = slice_python_object_as_numpy(sequence, valid_backbone_indices)
    atom_positions = atom_positions[valid_backbone_mask, ...]
    residue_index = residue_index[valid_backbone_mask]
    atom_mask = atom_mask[valid_backbone_mask, ...]
    confidence = confidence[valid_backbone_mask]
    aatype = np.array([RC.restype_1toidx[aa] for aa in sequence], dtype=np.int32)

    return aatype, atom_positions, atom_mask, residue_index, confidence


@dataclasses.dataclass
class Protein:
    """Protein structure representation."""

    id: str
    entity_id: int | None  # None for pdb file
    chain_id: str  # author chain id

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # residue_constants.atom_types, i.e. the first three are N, CA, CB.
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

    # Amino-acid type for each residue represented as an integer between 0 and
    # 20, where 20 is 'X'.
    aatype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB for pdb format. It is not necessarily continuous or 0-indexed.
    # equal to label_seq_id for cif format.
    residue_index: np.ndarray  # [num_res]

    # Resolution is 0 for AFDB proteins by default and 0 for PDB file by default
    resolution: float

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    # B-factor of CA atom for each residue
    b_factors: np.ndarray | None  # [num_res]

    # pLDDT
    # It is stored in the B-factor fields of the mmCIF and PDB files available for download (although unlike a B-factor, higher pLDDT is better).
    plddt: np.ndarray | None  # [num_res]

    @classmethod
    def from_afdb_tar_seek(
        cls, tar_file_path: str | Path, seek: int, entity_id: int, chain_id: str
    ) -> "Protein":
        with open(str(tar_file_path), "rb") as f:
            f.seek(seek)
            with tarfile.open(fileobj=f, mode="r|*") as tar:
                member = tar.extractfile(tar.next())
                with gzip.open(member, "rt") as file:
                    cif_file = pdbx.CIFFile.read(file)
        return cls.from_cif_file(
            cif_file=cif_file,
            entity_id=entity_id,
            chain_id=chain_id,
            type="afdb",
        )

    @classmethod
    def from_cif_file_path(
        cls, cif_file_path: str | Path, entity_id: int, chain_id: str
    ) -> "Protein":
        cif_file_path = Path(cif_file_path)
        if "gz" in cif_file_path.suffix:
            with gzip.open(str(cif_file_path), "rt") as file:
                cif_file = pdbx.CIFFile.read(file)
        else:
            cif_file = pdbx.CIFFile.read(str(cif_file_path))
        return cls.from_cif_file(
            cif_file=cif_file,
            entity_id=entity_id,
            chain_id=chain_id,
            type="exp",
        )

    @classmethod
    def from_pdb_file_path(cls, pdb_file_path: str | Path, chain_id: str) -> "Protein":
        pdb_file_path = Path(pdb_file_path)
        id = pdb_file_path.stem
        if "gz" in pdb_file_path.suffix:
            with gzip.open(str(pdb_file_path), "rt") as file:
                pdb_file = pdb.PDBFile.read(file)
        else:
            pdb_file = pdb.PDBFile.read(str(pdb_file_path))
        return cls.from_pdb_file(
            pdb_file=pdb_file,
            id=id,
            chain_id=chain_id,
        )

    @classmethod
    def from_pdb_file(cls, pdb_file: pdb.PDBFile, id: str, chain_id: str) -> "Protein":
        atom_array = pdb.get_structure(pdb_file, model=1, extra_fields=["b_factor"])
        if chain_id == "nan":
            atom_array = atom_array[
                bs.filter_amino_acids(atom_array) & ~atom_array.hetero
            ]
        else:
            atom_array = atom_array[
                bs.filter_amino_acids(atom_array)
                & ~atom_array.hetero
                & (atom_array.chain_id == chain_id)
            ]

        if atom_array.array_length() == 0:
            raise ProteinChainEmpty(f"id {id} chain_id {chain_id} has no valid atoms.")

        aatype, atom_positions, atom_mask, residue_index, confidence = (
            _process_atom_array(atom_array=atom_array)
        )

        if len(atom_positions) == 0:
            raise ProteinChainEmpty(f"id {id} chain_id {chain_id} has no valid atoms.")
        if len(atom_positions) < MIN_NB_RES:
            raise ProteinChainTooSmall(
                f"id {id} chain_id {chain_id} has {len(atom_positions)} residues which is below the minimum number {MIN_NB_RES}."
            )

        return cls(
            id=id,
            entity_id=None,
            resolution=0.0,
            chain_id=chain_id,
            residue_index=residue_index,
            atom_positions=atom_positions,
            atom_mask=atom_mask,
            aatype=aatype,
            b_factors=confidence,
            plddt=None,
        )

    @classmethod
    def from_cif_file(
        cls,
        cif_file: pdbx.CIFFile,
        entity_id: int,
        chain_id: str,
        type: Literal["afdb", "exp"],
    ) -> "Protein":
        id = list(cif_file.keys())[0]
        # resolution
        try:
            resolution = get_resolution(cif_file=cif_file)
        except Exception as e:
            resolution = 0.0
        atom_array = pdbx.get_structure(
            cif_file,
            model=1,
            extra_fields=["b_factor", "label_seq_id", "label_entity_id"],
        )
        atom_array = atom_array[
            bs.filter_amino_acids(atom_array)
            & ~atom_array.hetero
            & (atom_array.chain_id == chain_id)
            & (atom_array.label_entity_id == str(entity_id))
        ]

        if atom_array.array_length() == 0:
            raise ProteinChainEmpty(
                f"id {id} entity_id {str(entity_id)} chain_id {chain_id} has no valid atoms."
            )

        aatype, atom_positions, atom_mask, residue_index, confidence = (
            _process_atom_array(atom_array=atom_array)
        )

        if len(atom_positions) == 0:
            raise ProteinChainEmpty(
                f"id {id} entity_id {entity_id} chain_id {chain_id} has no valid atoms."
            )
        if len(atom_positions) < MIN_NB_RES:
            raise ProteinChainTooSmall(
                f"id {id} entity_id {entity_id} chain_id {chain_id} has {len(atom_positions)} residues which is below the minimum number {MIN_NB_RES}."
            )

        return cls(
            id=id,
            entity_id=entity_id,
            chain_id=chain_id,
            residue_index=residue_index,
            atom_positions=atom_positions,
            atom_mask=atom_mask,
            aatype=aatype,
            resolution=resolution,
            b_factors=confidence if type == "exp" else None,
            plddt=confidence if type == "afdb" else None,
        )

    def to_torch_input(self) -> dict[str, torch.Tensor | str | None]:
        return {
            "id": self.id,
            "entity_id": self.entity_id,
            "chain_id": self.chain_id,
            "resolution": torch.tensor(self.resolution, dtype=torch.float),
            "atom_positions": torch.from_numpy(self.atom_positions),
            "aatype": torch.from_numpy(self.aatype),
            "atom_mask": torch.from_numpy(self.atom_mask),
            "residue_index": torch.from_numpy(self.residue_index),
        }

    def __str__(self) -> str:
        return (
            "_".join([str(self.id), str(self.entity_id), str(self.chain_id)])
            if self.entity_id is not None
            else "_".join([str(self.id), str(self.chain_id)])
        )

    @cached_property
    def atom_array(self) -> bs.AtomArray:
        atoms = []
        nb_res = len(self.atom_positions)
        if self.b_factors is not None:
            confidence = self.b_factors
        elif self.plddt is not None:
            confidence = self.plddt
        else:
            confidence = [None for _ in range(nb_res)]
        for aa, res_idx, positions, mask, conf in zip(
            self.aatype,
            self.residue_index,
            self.atom_positions,
            self.atom_mask.astype(bool),
            confidence,
        ):
            res_name = RC.restype_1to3.get(RC.restype_idxto1.get(aa), "UNK")
            for i, pos in zip(np.where(mask)[0], positions[mask]):
                atom = bs.Atom(
                    coord=pos,
                    chain_id="" if self.chain_id == "nan" else self.chain_id,
                    res_id=res_idx,
                    res_name=res_name,
                    atom_name=RC.atom_types[i],
                    element=RC.atom_types[i][0],
                    b_factor=conf if conf is not None else 0.0,
                )
                atoms.append(atom)
        return bs.array(atoms)

    def to_pdb(self, path: PathLike) -> None:
        f = pdb.PDBFile()
        f.set_structure(self.atom_array)
        f.write(str(path))
