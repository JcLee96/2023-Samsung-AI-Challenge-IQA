"""Meshed-Memory Transformer (M2) captioning model, vendored from aimagelab/meshed-memory-transformer via RSTNet."""
from .transformer import Transformer
from .encoders import MemoryAugmentedEncoder
from .decoders import MeshedDecoder
from .attention import ScaledDotProductAttentionMemory
from .captioning_model import CaptioningModel
