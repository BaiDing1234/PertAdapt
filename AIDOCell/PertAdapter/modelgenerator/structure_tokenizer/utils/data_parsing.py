import gzip
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from biotite.structure.io import pdbx, pdb
import biotite.structure as bs


def register_mmcif_chains(cif_file_path: str | Path) -> pd.DataFrame:
    """
    Registers the chains in an mmCIF file and returns a DataFrame with metadata about each chain.

    Parameters:
    ----------
    cif_file_path : str | Path
        The path to the mmCIF file. Can be a string or a Path object.

    Returns:
    -------
    pl.DataFrame
        A DataFrame containing metadata about each chain in the mmCIF file. The DataFrame includes the following columns:
        - filename: The name of the mmCIF file.
        - pdb_id: The identifier of the mmCIF file (stem of the file name).
        - entity: The entity identifier.
        - chain: The chain identifier.
        - nb_residues: The number of residues in the chain.
        - nb_complete_backbones: The number of complete backbones in the chain.

    Notes:
    -----
    - The function supports both plain mmCIF files and gzipped mmCIF files.
    - Only amino acid chains are considered; hetero atoms are excluded.
    """
    cif_file_path = Path(cif_file_path)
    filename = cif_file_path.name
    if "gz" in cif_file_path.suffix:
        with gzip.open(str(cif_file_path), "rt") as file:
            cif_file = pdbx.CIFFile.read(file)
    else:
        cif_file = pdbx.CIFFile.read(str(cif_file_path))
    pdb_id = list(cif_file.keys())[0]
    atom_array = pdbx.get_structure(
        cif_file, model=1, extra_fields=["label_seq_id", "label_entity_id"]
    )
    atom_array = atom_array[bs.filter_amino_acids(atom_array) & ~atom_array.hetero]
    metadata = defaultdict(list)
    for atom_array_chain in bs.chain_iter(atom_array):
        entity = atom_array_chain.label_entity_id.astype(int)[0].item()
        chain_id = atom_array_chain.chain_id[0]
        nb_residues = bs.get_residue_count(atom_array_chain)
        backbone_atoms = bs.filter_peptide_backbone(atom_array_chain)
        nb_complete_backbones = (
            np.diff(
                bs.get_residue_starts(atom_array_chain[backbone_atoms]),
                append=backbone_atoms.sum(),
            )
            == 3
        ).sum()

        metadata["filename"].append(filename)
        metadata["pdb_id"].append(pdb_id)
        metadata["entity"].append(entity)
        metadata["chain"].append(chain_id)
        metadata["nb_residues"].append(nb_residues)
        metadata["nb_complete_backbones"].append(nb_complete_backbones)

    df = pd.DataFrame(metadata).astype(
        {
            "filename": str,
            "pdb_id": str,
            "entity": "uint16",
            "chain": str,
            "nb_residues": "uint16",
            "nb_complete_backbones": "uint16",
        }
    )
    return df


def register_pdb_chains(pdb_file_path: str | Path) -> pd.DataFrame:
    """
    Registers the chains in a PDB file and returns a DataFrame with metadata about each chain.

    Parameters:
    ----------
    pdb_file_path : str | Path
        The path to the PDB file. Can be a string or a Path object.

    Returns:
    -------
    pl.DataFrame
        A DataFrame containing metadata about each chain in the PDB file. The DataFrame includes the following columns:
        - filename: The name of the PDB file.
        - pdb_id: The identifier of the PDB file (stem of the file name).
        - chain: The chain identifier.
        - nb_residues: The number of residues in the chain.
        - nb_complete_backbones: The number of complete backbones in the chain.

    Notes:
    -----
    - The function supports both plain PDB files and gzipped PDB files.
    - Only amino acid chains are considered; hetero atoms are excluded.
    - A complete backbone is defined as N, CA, C
    """
    pdb_file_path = Path(pdb_file_path)
    filename = pdb_file_path.name
    if "gz" in pdb_file_path.suffix:  # ent.gz file
        with gzip.open(str(pdb_file_path), "rt") as file:
            pdb_file = pdb.PDBFile.read(file)
    else:
        pdb_file = pdb.PDBFile.read(str(pdb_file_path))
    pdb_id = pdb_file_path.stem
    atom_array = pdb.get_structure(pdb_file, model=1)
    atom_array = atom_array[bs.filter_amino_acids(atom_array) & ~atom_array.hetero]
    metadata = defaultdict(list)
    for atom_array_chain in bs.chain_iter(atom_array):
        chain_id = atom_array_chain.chain_id[0].item()
        nb_residues = bs.get_residue_count(atom_array_chain)
        backbone_atoms = bs.filter_peptide_backbone(atom_array_chain)
        nb_complete_backbones = (
            np.diff(
                bs.get_residue_starts(atom_array_chain[backbone_atoms]),
                append=backbone_atoms.sum(),
            )
            == 3
        ).sum()

        metadata["filename"].append(filename)
        metadata["pdb_id"].append(pdb_id)
        metadata["chain"].append(chain_id)
        metadata["nb_residues"].append(nb_residues)
        metadata["nb_complete_backbones"].append(nb_complete_backbones)

    df = pd.DataFrame(metadata).astype(
        {
            "filename": str,
            "pdb_id": str,
            "chain": str,
            "nb_residues": "uint16",
            "nb_complete_backbones": "uint16",
        }
    )
    return df
