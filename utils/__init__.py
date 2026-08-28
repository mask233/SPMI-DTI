"""Training utilities with lazy public imports.

Importing a lightweight submodule such as ``utils.split_manifest`` must not
eagerly import pandas, RDKit, torch, and PyG.  The training-facing names remain
backward compatible and are loaded only when requested.
"""

__all__ = ["protein_init", "ligand_init", "smiles2graph", "ProteinMoleculeDataset"]


def __getattr__(name):
    if name == "protein_init":
        from .protein_init import protein_init
        return protein_init
    if name in {"ligand_init", "smiles2graph"}:
        from .ligand_init import ligand_init, smiles2graph
        return {"ligand_init": ligand_init, "smiles2graph": smiles2graph}[name]
    if name == "ProteinMoleculeDataset":
        from .dataset import ProteinMoleculeDataset
        return ProteinMoleculeDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
