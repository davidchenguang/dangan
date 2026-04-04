"""transformers 4.57+ 兼容性补丁

从 test_ocr.py:18-91 迁移，必须在新版本 transformers 上执行。

修复内容:
1. LlamaFlashAttention2 / LlamaSdpaAttention 已移除 → 回退到 LlamaAttention
2. DynamicCache.seen_tokens / get_usable_length / get_max_length 已移除 → 属性注入
3. LlamaAttention.forward() 签名变更 (4.57+ 要求 position_embeddings)
"""

from __future__ import annotations

import inspect
import logging

logger = logging.getLogger(__name__)

_patches_applied = False


def apply_patches() -> None:
    """应用所有兼容性补丁（幂等，只执行一次）"""
    global _patches_applied
    if _patches_applied:
        return

    import transformers
    logger.info("检测到 transformers %s，开始应用兼容性补丁...", transformers.__version__)

    # 修复 1: LlamaFlashAttention2 / LlamaSdpaAttention 已移除
    _fix_llama_attention_aliases()

    # 修复 2: DynamicCache 属性缺失
    _fix_dynamic_cache()

    # 修复 3: LlamaAttention.forward() 签名变更
    _fix_llama_attention_forward()

    _patches_applied = True
    logger.info("兼容性补丁应用完成")


def _fix_llama_attention_aliases() -> None:
    """LlamaFlashAttention2 / LlamaSdpaAttention 在 transformers 4.57+ 中被移除"""
    try:
        import transformers.models.llama.modeling_llama as llama_module
    except ImportError:
        return

    if not hasattr(llama_module, "LlamaFlashAttention2"):
        if hasattr(llama_module, "LlamaAttention"):
            llama_module.LlamaFlashAttention2 = llama_module.LlamaAttention
            logger.debug("已注入 LlamaFlashAttention2 别名")

    if not hasattr(llama_module, "LlamaSdpaAttention"):
        if hasattr(llama_module, "LlamaAttention"):
            llama_module.LlamaSdpaAttention = llama_module.LlamaAttention
            logger.debug("已注入 LlamaSdpaAttention 别名")


def _fix_dynamic_cache() -> None:
    """DynamicCache 在 transformers 4.57+ 中移除了 seen_tokens 等属性"""
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        return

    if not hasattr(DynamicCache, 'seen_tokens'):
        DynamicCache.seen_tokens = property(
            lambda self: self._seen_tokens
            if hasattr(self, '_seen_tokens')
            else self.get_seq_length()
        )
        logger.debug("已注入 DynamicCache.seen_tokens 属性")

    if not hasattr(DynamicCache, 'get_usable_length'):
        DynamicCache.get_usable_length = lambda self, *a, **kw: self.get_seq_length()
        logger.debug("已注入 DynamicCache.get_usable_length 方法")

    if not hasattr(DynamicCache, 'get_max_length'):
        DynamicCache.get_max_length = lambda self: None
        logger.debug("已注入 DynamicCache.get_max_length 方法")


def _fix_llama_attention_forward() -> None:
    """LlamaAttention.forward() 签名变更: position_ids → position_embeddings"""
    try:
        import transformers.models.llama.modeling_llama as llama_module
        import torch
    except ImportError:
        return

    _orig_forward = llama_module.LlamaAttention.forward
    _orig_params = list(inspect.signature(_orig_forward).parameters.keys())

    # 仅在签名包含 position_embeddings 时才需要补丁
    if "position_embeddings" not in _orig_params:
        return

    from transformers.models.llama.modeling_llama import (
        LlamaRotaryEmbedding,
        apply_rotary_pos_emb,
        repeat_kv,
    )

    def _compat_forward(
        self, hidden_states, attention_mask=None, position_ids=None,
        past_key_value=None, output_attentions=False, use_cache=False,
        cache_position=None, **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()
        head_dim = getattr(
            self.config, "head_dim",
            self.config.hidden_size // self.config.num_attention_heads,
        )
        hidden_shape = (bsz, q_len, self.config.num_attention_heads, head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if hasattr(self, 'rotary_emb') and position_ids is not None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = None, None

        if cos is not None and sin is not None:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs,
            )

        n_rep = self.config.num_attention_heads // self.config.num_key_value_heads
        key_states = repeat_kv(key_states, n_rep)
        value_states = repeat_kv(value_states, n_rep)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (head_dim ** 0.5)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32,
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    llama_module.LlamaAttention.forward = _compat_forward
    logger.debug("已替换 LlamaAttention.forward 为兼容版本")


def inject_rotary_embeddings(model) -> None:
    """给使用 LlamaAttention 的层注入 rotary_emb（修复 3 的补充步骤）

    在模型加载后调用一次。
    """
    try:
        import transformers.models.llama.modeling_llama as llama_module
    except ImportError:
        return

    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding

    count = 0
    for _name, module in model.named_modules():
        if isinstance(module, LlamaAttention) and not hasattr(module, 'rotary_emb'):
            module.rotary_emb = LlamaRotaryEmbedding(config=module.config, device=model.device)
            count += 1

    if count > 0:
        logger.info("已为 %d 个 LlamaAttention 层注入 rotary_emb", count)
