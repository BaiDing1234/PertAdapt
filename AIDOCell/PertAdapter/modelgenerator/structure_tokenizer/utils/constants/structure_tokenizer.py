from modelgenerator.structure_tokenizer.utils.constants import residue_constants

SCALE_POSITIONS: float = 10
QUANTIZE_IDX_MASK: int = (
    -100
)  # -100 is a special value that will be ignored in the loss function

DISTOGRAM_BINS: int = 64
LDDT_BINS: int = 50

# Tokens to predict residues
# 0 is padding, N + 1 is mask.
N_TOKENS = residue_constants.restype_num + 2
PAD_IDX = 0
UNK_IDX = N_TOKENS - 2
MASK_IDX = N_TOKENS - 1
