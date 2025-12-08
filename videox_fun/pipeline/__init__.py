from .pipeline_wan import WanPipeline
from .pipeline_wan2_2 import Wan2_2Pipeline

WanFunPipeline = WanPipeline
Wan2_2FunPipeline = Wan2_2Pipeline

import importlib.util

if importlib.util.find_spec("paifuser") is not None:
    # --------------------------------------------------------------- #
    #   Sparse Attention
    # --------------------------------------------------------------- #
    from paifuser.ops import sparse_reset

    # Wan2.1
    WanFunPipeline.__call__ = sparse_reset(WanFunPipeline.__call__)
    WanPipeline.__call__ = sparse_reset(WanPipeline.__call__)

    # Wan2.2
    Wan2_2FunPipeline.__call__ = sparse_reset(Wan2_2FunPipeline.__call__)
    Wan2_2Pipeline.__call__ = sparse_reset(Wan2_2Pipeline.__call__)
