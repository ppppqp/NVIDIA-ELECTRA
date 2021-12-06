from configuration_utils import PretrainedConfig
import tensorflow as tf
class ElectraConfig(PretrainedConfig):
    model_type = "ganzs"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        # GPT-2 config
        vocab_size=30522,
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=30522,
        eos_token_id=30522,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd

        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = reorder_and_upcast_attn

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        ## synonym
        self.max_position_embeddings = self.n_positions
        self.num_hidden_layers = self.n_layer
        self.num_attention_heads = self.n_head
        self.hidden_size = self.n_embd
        self.embedding_size = self.n_embd
        self.type_vocab_size = 2
        self.layer_norm_eps = self.layer_norm_epsilon
        self.hidden_dropout_prob = self.embd_pdrop
        self.hidden_act=self.activation_function
        # some extra
        self.add_cross_attention = False

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)



