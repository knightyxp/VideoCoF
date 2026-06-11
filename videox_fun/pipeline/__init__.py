from .pipeline_wan import WanPipeline

WanFunPipeline = WanPipeline

import importlib.util

if importlib.util.find_spec("paifuser") is not None:
    # --------------------------------------------------------------- #
    #   Sparse Attention
    # --------------------------------------------------------------- #
    from paifuser.ops import sparse_reset

    # Wan2.1
    WanFunPipeline.__call__ = sparse_reset(WanFunPipeline.__call__)
    WanPipeline.__call__ = sparse_reset(WanPipeline.__call__)
