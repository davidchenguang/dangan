"""
兼容性修复核心模块：处理 transformers 库版本差异
专门处理 4.57+ 版本引起的 API 不兼容问题，避免污染业务逻辑。
"""

import inspect
import logging

logger = logging.getLogger(__name__)

_patched = False

def apply_patches():
    global _patched
    if _patched:
        return
        
    try:
        import transformers.models.llama.modeling_llama as llama_module
    except ImportError:
        logger.warning("transformers 未安装或模块错误，放弃修补。")
        return

    # 修复 1: LlamaFlashAttention2 / LlamaSdpaAttention 已移除
    if not hasattr(llama_module, "LlamaFlashAttention2"):
        if hasattr(llama_module, "LlamaAttention"):
            llama_module.LlamaFlashAttention2 = llama_module.LlamaAttention
            llama_module.LlamaSdpaAttention = llama_module.LlamaAttention
    if not hasattr(llama_module, "LlamaSdpaAttention"):
        if hasattr(llama_module, "LlamaAttention"):
            llama_module.LlamaSdpaAttention = llama_module.LlamaAttention

    # 修复 2: DynamicCache seen_tokens / get_usable_length / get_max_length 已移除
    try:
        from transformers.cache_utils import DynamicCache
        if not hasattr(DynamicCache, 'seen_tokens'):
            DynamicCache.seen_tokens = property(
                lambda self: self._seen_tokens
                if hasattr(self, '_seen_tokens')
                else self.get_seq_length()
            )
        if not hasattr(DynamicCache, 'get_usable_length'):
            DynamicCache.get_usable_length = lambda self, *a, **kw: self.get_seq_length()
        if not hasattr(DynamicCache, 'get_max_length'):
            DynamicCache.get_max_length = lambda self: None
    except ImportError:
        pass

    # 修复 3: LlamaAttention.forward() 签名变更 (4.57+ 要求 position_embeddings 而非 position_ids)
    try:
        _orig_llama_attn_forward = llama_module.LlamaAttention.forward
        _orig_params = list(inspect.signature(_orig_llama_attn_forward).parameters.keys())

        if "position_embeddings" in _orig_params:
            from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb, repeat_kv

            def _compat_forward(
                self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, **kwargs,
            ):
                import torch
                bsz, q_len, _ = hidden_states.size()
                head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
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
                    key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

                n_rep = self.config.num_attention_heads // self.config.num_key_value_heads
                key_states = repeat_kv(key_states, n_rep)
                value_states = repeat_kv(value_states, n_rep)

                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (head_dim ** 0.5)

                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask

                attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_output = torch.matmul(attn_weights, value_states)
                attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
                attn_output = self.o_proj(attn_output)

                return attn_output, None, past_key_value

            llama_module.LlamaAttention.forward = _compat_forward
    except Exception as e:
        logger.warning(f"修补 LlamaAttention 失败: {e}")
        
    _patched = True
