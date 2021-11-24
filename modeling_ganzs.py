# Copyright 2020 The Google AI Team, Stanford University and The HuggingFace Inc. team.
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import tensorflow as tf
# from transformers import TFGPT2MainLayer, TFBlock, TFBaseModelOutputWithPastAndCrossAttentions
# from transformers import TFGPT2PreTrainedModel
from modeling_utils import TFBlock
from configuration_ganzs import ElectraConfig
from file_utils import add_start_docstrings, add_start_docstrings_to_callable
from modeling_utils import ACT2FN, TFBertEncoder, TFGPT2PreTrainedModel
from modeling_utils import get_initializer, shape_list
from tokenization_utils import BatchEncoding
import pretrain_utils, collections

logger = logging.getLogger(__name__)


TF_ELECTRA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "google/electra-small-generator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-small-generator/tf_model.h5",
    "google/electra-base-generator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-base-generator/tf_model.h5",
    "google/electra-large-generator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-large-generator/tf_model.h5",
    "google/electra-small-discriminator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-small-discriminator/tf_model.h5",
    "google/electra-base-discriminator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-base-discriminator/tf_model.h5",
    "google/electra-large-discriminator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-large-discriminator/tf_model.h5",
}


class TFElectraEmbeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.initializer_range = config.initializer_range

        self.position_embeddings = tf.keras.layers.Embedding(
            config.max_position_embeddings,
            config.embedding_size,
            embeddings_initializer=get_initializer(self.initializer_range),
            name="position_embeddings",
        )
        self.token_type_embeddings = tf.keras.layers.Embedding(
            config.type_vocab_size,
            config.embedding_size,
            embeddings_initializer=get_initializer(self.initializer_range),
            name="token_type_embeddings",
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.amp = config.amp

    def build(self, input_shape):
        """Build shared word embedding layer """
        with tf.name_scope("word_embeddings"):
            # Create and initialize weights. The random normal initializer was chosen
            # arbitrarily, and works well.
            self.word_embeddings = self.add_weight(
                "weight",
                shape=[self.vocab_size, self.embedding_size],
                initializer=get_initializer(self.initializer_range),
            )
        super().build(input_shape)

    def call(self, inputs, mode="embedding", training=False):
        """Get token embeddings of inputs.
        Args:
            inputs: list of three int64 tensors with shape [batch_size, length]: (input_ids, position_ids, token_type_ids)
            mode: string, a valid value is one of "embedding" and "linear".
        Returns:
            outputs: (1) If mode == "embedding", output embedding tensor, float32 with
                shape [batch_size, length, embedding_size]; (2) mode == "linear", output
                linear tensor, float32 with shape [batch_size, length, vocab_size].
        Raises:
            ValueError: if mode is not valid.

        Shared weights logic adapted from
            https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24
        """
        if mode == "embedding":
            return self._embedding(inputs, training=training)
        elif mode == "linear":
            return self._linear(inputs)
        else:
            raise ValueError("mode {} is not valid.".format(mode))

    def _embedding(self, inputs, training=False):
        """Applies embedding based on inputs tensor."""
        input_ids, position_ids, token_type_ids, inputs_embeds = inputs

        if input_ids is not None:
            input_shape = shape_list(input_ids)
        else:
            input_shape = shape_list(inputs_embeds)[:-1]

        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = tf.range(seq_length, dtype=tf.int32)[tf.newaxis, :]
        if token_type_ids is None:
            token_type_ids = tf.fill(input_shape, 0)

        if inputs_embeds is None:
            inputs_embeds = tf.gather(self.word_embeddings, input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        if self.amp:
            embeddings = inputs_embeds + tf.cast(position_embeddings, tf.float16) + tf.cast(token_type_embeddings, tf.float16)
        else:
            embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings, training=training)
        return embeddings

    def _linear(self, inputs):
        """Computes logits by running inputs through a linear layer.
            Args:
                inputs: A float32 tensor with shape [batch_size, length, hidden_size]
            Returns:
                float32 tensor with shape [batch_size, length, vocab_size].
        """
        batch_size = shape_list(inputs)[0]
        length = shape_list(inputs)[1]

        x = tf.reshape(inputs, [-1, self.embedding_size])
        logits = tf.matmul(x, self.word_embeddings, transpose_b=True)

        return tf.reshape(logits, [batch_size, length, self.vocab_size])


#DONE
class TFElectraDiscriminatorPredictions(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense")
        self.dense_prediction = tf.keras.layers.Dense(
            1, kernel_initializer=get_initializer(config.initializer_range), name="dense_prediction")
        self.config = config

    def call(self, discriminator_hidden_states, training=False):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = ACT2FN[self.config.hidden_act](hidden_states)
        logits = tf.squeeze(self.dense_prediction(hidden_states), axis=-1)

        return logits


class TFElectraGeneratorPredictions(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dense = tf.keras.layers.Dense(
            config.embedding_size, kernel_initializer=get_initializer(config.initializer_range), name="dense")

    def call(self, generator_hidden_states, training=False):
        hidden_states = self.dense(generator_hidden_states)
        hidden_states = ACT2FN["gelu"](hidden_states)
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states


class TFElectraPreTrainedModel(TFGPT2PreTrainedModel):

    config_class = ElectraConfig
    pretrained_model_archive_map = TF_ELECTRA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "electra"

    def get_extended_attention_mask(self, attention_mask, input_shape):
        if attention_mask is None:
            attention_mask = tf.fill(input_shape, 1)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.

        extended_attention_mask = tf.cast(extended_attention_mask, tf.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask

    def get_head_mask(self, head_mask):
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.config.num_hidden_layers

        return head_mask

# SWAP WITH TFGPT2MainLayer
class TFElectraMainLayer(TFElectraPreTrainedModel):

    config_class = ElectraConfig

    def __init__(self, config, shared_embeddings=False, input_embeddings=None, **kwargs):
        super().__init__(config, **kwargs)

        if shared_embeddings and input_embeddings is not None:
            self.wte = input_embeddings
        else:
            self.wte = TFElectraEmbeddings(config, name="embeddings")
        #synonym
        self.embeddings = self.wte
        if config.embedding_size != config.hidden_size:
            self.embeddings_project = tf.keras.layers.Dense(
                config.hidden_size,
                kernel_initializer=get_initializer(config.initializer_range),
                name="embeddings_project")
        # self.encoder = TFGPT2Model(config, name="encoder")
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        # self.use_cache = config.use_cache
        # self.return_dict = config.use_return_dict

        self.num_hidden_layers = config.n_layer
        self.vocab_size = config.vocab_size
        self.n_embd = config.n_embd
        self.n_positions = config.n_positions
        self.initializer_range = config.initializer_range
        self.drop = tf.keras.layers.Dropout(config.embd_pdrop)
        self.h = [TFBlock(config, scale=True, name=f"h_._{i}") for i in range(config.n_layer)]
        self.ln_f = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="ln_f")


    def get_input_embeddings(self):
        return self.embeddings

    def _resize_token_embeddings(self, new_num_tokens):
        raise NotImplementedError

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        raise NotImplementedError

    def call(
        self,
        inputs,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
        **kwargs,
    ):
        if isinstance(inputs, (tuple, list)):
            # FIXME!
            input_ids = inputs[0]
            past_key_values = inputs[1] if len(inputs) > 1 else past_key_values
            attention_mask = inputs[2] if len(inputs) > 2 else attention_mask
            token_type_ids = inputs[3] if len(inputs) > 3 else token_type_ids
            position_ids = inputs[4] if len(inputs) > 4 else position_ids
            head_mask=inputs[5] if len(inputs) > 5 else position_ids
            inputs_embeds = inputs[6] if len(inputs) > 6 else inputs_embeds
            encoder_hidden_states = inputs[7] if len(inputs) > 7 else encoder_hidden_states
            encoder_attention_mask = inputs[8] if len(inputs) > 8 else encoder_attention_mask
            use_cache = inputs[9] if len(inputs) > 9 else use_cache
            output_attentions = inputs[10] if len(inputs) > 10 else output_attentions
            output_hidden_states = inputs[11] if len(inputs) > 11 else output_hidden_states
            return_dict = inputs[12] if len(inputs) > 12 else return_dict

            assert len(inputs) <= 13, "Too many inputs."
        elif isinstance(inputs, (dict, BatchEncoding)):
            # FIXME!
            input_ids = inputs.get("input_ids")
            past_key_values = inputs.get("past_key_values", attention_mask)
            attention_mask = inputs.get("attention_mask", attention_mask)
            token_type_ids = inputs.get("token_type_ids", token_type_ids)
            position_ids = inputs.get("position_ids", position_ids)
            head_mask = inputs.get("head_mask", head_mask)
            inputs_embeds = inputs.get("inputs_embeds", inputs_embeds)
            encoder_hidden_states = inputs.get("encoder_hidden_states", encoder_hidden_states)
            encoder_attention_mask = inputs.get("encoder_hidden_mask", encoder_attention_mask)
            use_cache = inputs.get("use_cache", use_cache)
            output_attentions = inputs.get("output_attentions", output_attentions)
            output_hidden_states = inputs.get("output_hidden_states", output_hidden_states)
            return_dict = inputs.get("return_dict", return_dict)
            assert len(inputs) <= 10, "Too many inputs."
        else:
            input_ids = inputs

        if input_ids is not None and input_ids is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
            input_ids = tf.reshape(input_ids, [-1, input_shape[-1]])
        elif input_ids is not None:
            input_shape = shape_list(input_ids)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * len(self.h)
        else:
            past_length = shape_list(past_key_values[0][0])[-2]

        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(past_length, input_shape[-1] + past_length), axis=0)

        if attention_mask is not None:
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask_shape = shape_list(attention_mask)
            attention_mask = tf.reshape(
                attention_mask, (attention_mask_shape[0], 1, 1, attention_mask_shape[1])
            )

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            one_cst = tf.constant(1.0)
            attention_mask = tf.cast(attention_mask, dtype=one_cst.dtype)
            attention_mask = tf.multiply(
                tf.subtract(one_cst, attention_mask), tf.constant(-10000.0)
            )

        # Copied from `modeling_tf_t5.py` with -1e9 -> -10000
        if self.config.add_cross_attention and encoder_attention_mask is not None:
            # If a 2D ou 3D attention mask is provided for the cross-attention
            # we need to make broadcastable to [batch_size, num_heads, mask_seq_length, mask_seq_length]
            # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
            encoder_attention_mask = tf.cast(
                encoder_attention_mask, dtype=encoder_hidden_states.dtype
            )
            num_dims_encoder_attention_mask = len(shape_list(encoder_attention_mask))
            if num_dims_encoder_attention_mask == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            if num_dims_encoder_attention_mask == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

            # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
            # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow/transformer/transformer_layers.py#L270
            # encoder_extended_attention_mask = tf.math.equal(encoder_extended_attention_mask,
            #                                         tf.transpose(encoder_extended_attention_mask, perm=(-1, -2)))

            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        encoder_attention_mask = encoder_extended_attention_mask

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask= [None] * self.num_hidden_layers
            # head_mask = tf.constant([0] * self.num_hidden_layers)

        position_ids = tf.reshape(position_ids, [-1, shape_list(position_ids)[-1]])

        if input_ids is None:
            input_ids = self.wte(input_ids, mode="embedding")

        position_embeds = tf.gather(self.wpe, position_ids)

        if token_type_ids is not None:
            token_type_ids = tf.reshape(
                token_type_ids, [-1, shape_list(token_type_ids)[-1]]
            )
            token_type_embeds = self.wte(token_type_ids, mode="embedding")
        else:
            token_type_embeds = tf.constant(0.0)

        position_embeds = tf.cast(position_embeds, dtype=input_ids.dtype)
        token_type_embeds = tf.cast(token_type_embeds, dtype=input_ids.dtype)
        hidden_states = input_ids + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states, training=training)

        output_shape = input_shape + [shape_list(hidden_states)[-1]]

        presents = () if use_cache else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (tf.reshape(hidden_states, output_shape),)

            outputs = block(
                hidden_states,
                layer_past,
                attention_mask,
                head_mask[i],
                encoder_hidden_states,
                encoder_attention_mask,
                use_cache,
                output_attentions,
                training=training,
            )

            hidden_states, present = outputs[:2]
            if use_cache:
                presents = presents + (present,)

            if output_attentions:
                all_attentions = all_attentions + (outputs[2],)
                if self.config.add_cross_attention and encoder_hidden_states is not None:
                    all_cross_attentions = all_cross_attentions + (outputs[3],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = tf.reshape(hidden_states, output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + [-1] + shape_list(all_attentions[0])[-2:]
            all_attentions = tuple(tf.reshape(t, attention_output_shape) for t in all_attentions)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_attentions, all_cross_attentions]
                if v is not None
            )

        return TFBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


ELECTRA_START_DOCSTRING = r"""
    This model is a `tf.keras.Model <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`__ sub-class.
    Use it as a regular TF 2.0 Keras Model and
    refer to the TF 2.0 documentation for all matter related to general usage and behavior.

    .. note::

        TF 2.0 models accepts two formats as inputs:

            - having all inputs as keyword arguments (like PyTorch models), or
            - having all inputs as a list, tuple or dict in the first positional arguments.

        This second option is useful when using :obj:`tf.keras.Model.fit()` method which currently requires having
        all the tensors in the first argument of the model call function: :obj:`model(inputs)`.

        If you choose this second option, there are three possibilities you can use to gather all the input Tensors
        in the first positional argument :

        - a single Tensor with input_ids only and nothing else: :obj:`model(inputs_ids)`
        - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
          :obj:`model([input_ids, attention_mask])` or :obj:`model([input_ids, attention_mask, token_type_ids])`
        - a dictionary with one or several input Tensors associated to the input names given in the docstring:
          :obj:`model({'input_ids': input_ids, 'token_type_ids': token_type_ids})`

    Parameters:
        config (:class:`~transformers.ElectraConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

ELECTRA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.ElectraTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.encode_plus` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

            `What are attention masks? <../glossary.html#attention-mask>`__
        head_mask (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        inputs_embeds (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, embedding_dim)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        training (:obj:`boolean`, `optional`, defaults to :obj:`False`):
            Whether to activate dropout modules (if set to :obj:`True`) during training or to de-activate them
            (if set to :obj:`False`) for evaluation.

"""


@add_start_docstrings(
    "The bare Electra Model transformer outputting raw hidden-states without any specific head on top. Identical to "
    "the BERT model except that it uses an additional linear layer between the embedding layer and the encoder if the "
    "hidden size and embedding size are different."
    ""
    "Both the generator and discriminator checkpoints may be loaded into this model.",
    ELECTRA_START_DOCSTRING,
)
class TFElectraModel(TFElectraPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.electra = TFElectraMainLayer(config, name="electra")

    def get_input_embeddings(self):
        return self.electra.embeddings

    @add_start_docstrings_to_callable(ELECTRA_INPUTS_DOCSTRING)
    def call(self, inputs, **kwargs):
        r"""
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.ElectraConfig`) and inputs:
        last_hidden_state (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when :obj:`config.output_hidden_states=True`):
            tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``config.output_attentions=True``):
            tuple of :obj:`tf.Tensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`:

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import tensorflow as tf
        from transformers import ElectraTokenizer, TFElectraModel

        tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
        model = TFElectraModel.from_pretrained('google/electra-small-discriminator')
        input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        """
        outputs = self.electra(inputs, **kwargs)
        return outputs


@add_start_docstrings(
    """
Electra model with a binary classification head on top as used during pre-training for identifying generated
tokens.

Even though both the discriminator and generator may be loaded into this model, the discriminator is
the only model of the two to have the correct classification head to be used for this model.""",
    ELECTRA_START_DOCSTRING,
)
# DONE
class TFElectraForPreTraining(TFElectraPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.electra = TFElectraMainLayer(config)
        self.discriminator_predictions = TFElectraDiscriminatorPredictions(config, name="discriminator_predictions")

    def get_input_embeddings(self):
        return self.electra.embeddings

    @add_start_docstrings_to_callable(ELECTRA_INPUTS_DOCSTRING)
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        training=False,
    ):
        r"""
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.ElectraConfig`) and inputs:
        scores (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`):
            Prediction scores of the head (scores for each token before SoftMax).
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when :obj:`config.output_hidden_states=True`):
            tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``config.output_attentions=True``):
            tuple of :obj:`tf.Tensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`:

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import tensorflow as tf
        from transformers import ElectraTokenizer, TFElectraForPreTraining

        tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
        model = TFElectraForPreTraining.from_pretrained('google/electra-small-discriminator')
        input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
        outputs = model(input_ids)
        scores = outputs[0]
        """

        discriminator_hidden_states = self.electra(
            input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, training=training
        )
        discriminator_sequence_output = discriminator_hidden_states[0]
        logits = self.discriminator_predictions(discriminator_sequence_output)
        output = (logits,)
        output += discriminator_hidden_states[1:]

        return output  # (loss), scores, (hidden_states), (attentions)


class TFElectraMaskedLMHead(tf.keras.layers.Layer):
    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = config.vocab_size
        self.input_embeddings = input_embeddings

    def build(self, input_shape):
        self.bias = self.add_weight(shape=(self.vocab_size,), initializer="zeros", trainable=True, name="bias")
        super().build(input_shape)

    def call(self, hidden_states, training=False):
        hidden_states = self.input_embeddings(hidden_states, mode="linear")
        hidden_states = hidden_states + self.bias
        return hidden_states


@add_start_docstrings(
    """
Electra model with a language modeling head on top.

Even though both the discriminator and generator may be loaded into this model, the generator is
the only model of the two to have been trained for the masked language modeling task.""",
    ELECTRA_START_DOCSTRING,
)
class TFElectraForMaskedLM(TFElectraPreTrainedModel):
    def __init__(self, config, shared_embeddings=False, input_embeddings=None, **kwargs):
        super().__init__(config, **kwargs)

        self.vocab_size = config.vocab_size
        self.electra = TFElectraMainLayer(config,
                                          shared_embeddings=shared_embeddings,
                                          input_embeddings=input_embeddings,
                                          name="electra")
        self.generator_predictions = TFElectraGeneratorPredictions(config, name="generator_predictions")
        if isinstance(config.hidden_act, str):
            self.activation = ACT2FN[config.hidden_act]
        else:
            self.activation = config.hidden_act
        self.generator_lm_head = TFElectraMaskedLMHead(config, self.electra.embeddings, name="generator_lm_head")

    def get_input_embeddings(self):
        return self.electra.embeddings

    # def get_output_embeddings(self):
    #     return self.generator_lm_head

    @add_start_docstrings_to_callable(ELECTRA_INPUTS_DOCSTRING)
    def call(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
        **kwargs,
    ):
        r"""
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.ElectraConfig`) and inputs:
        prediction_scores (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when :obj:`config.output_hidden_states=True`):
            tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``config.output_attentions=True``):
            tuple of :obj:`tf.Tensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`:

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import tensorflow as tf
        from transformers import ElectraTokenizer, TFElectraForMaskedLM

        tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-generator')
        model = TFElectraForMaskedLM.from_pretrained('google/electra-small-generator')
        input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
        outputs = model(input_ids)
        prediction_scores = outputs[0]

        """
        generator_hidden_states = self.electra(
            input_ids=input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # missing some params
        generator_sequence_output = generator_hidden_states[0]
        prediction_scores = self.generator_predictions(generator_sequence_output, training=training)
        prediction_scores = self.generator_lm_head(prediction_scores, training=training)
        output = (prediction_scores,)
        output += generator_hidden_states[1:]

        return output  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)

def get_generator_config(config, bert_config):
    """Get model config for the generator network."""
    gen_config = ElectraConfig.from_dict(bert_config.to_dict())
    gen_config.hidden_size = int(round(
        bert_config.hidden_size * config.generator_hidden_size))
    #To keep hidden size divisble by 64 - attention head size
    if gen_config.hidden_size % 64 != 0:
        gen_config.hidden_size += 64 - (gen_config.hidden_size % 64)
    gen_config.num_hidden_layers = int(round(
        bert_config.num_hidden_layers * config.generator_layers))
    gen_config.intermediate_size = 4 * gen_config.hidden_size
    gen_config.num_attention_heads = max(1, gen_config.hidden_size // 64)
    return gen_config

class PretrainingModel(tf.keras.Model):
    """Transformer pre-training using the replaced-token-detection task."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # Set up model config
        self._config = config
        self.disc_config = ElectraConfig(
            vocab_size=config.vocab_size,
            n_positions=config.n_positions,
            # n_ctx=config.n_ctx,
            n_embd=config.n_embd,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_inner=config.n_inner,
            activation_function=config.activation_function,
            resid_pdrop=config.resid_pdrop,
            embd_pdrop=config.embd_pdrop,
            attn_pdrop=config.attn_pdrop,
            layer_norm_epsilon=config.layer_norm_epsilon,
            initializer_range=config.initializer_range,
            scale_attn_weights=config.scale_attn_weights,
            use_cache=config.use_cache
        )
        self.disc_config.update({"amp": config.amp})

        # Set up discriminator
        self.discriminator = TFElectraForPreTraining(self.disc_config)

        # Set up generator
        gen_config = get_generator_config(config, self.disc_config)
        gen_config.update({"amp": config.amp})
        if config.electra_objective:
            if config.shared_embeddings:
                self.generator = TFElectraForMaskedLM(
                    gen_config, shared_embeddings=True,
                    input_embeddings=self.discriminator.get_input_embeddings())
            else:
                self.generator = TFElectraForMaskedLM(gen_config)
        else:
            self.generator = TFElectraForMaskedLM(self.disc_config)

    def call(self, features, is_training):
        config = self._config

        # Mask the input
        masked_inputs = pretrain_utils.mask(
            config, pretrain_utils.features_to_inputs(features), config.mask_prob)

        # Generator
        if config.uniform_generator:
            mlm_output = self._get_masked_lm_output(masked_inputs, None, is_training=is_training)
        else:
            mlm_output = self._get_masked_lm_output(
                masked_inputs, self.generator, is_training=is_training)
        fake_data = self._get_fake_data(masked_inputs, mlm_output.logits)
        total_loss = config.gen_weight * mlm_output.loss

        # Discriminator
        disc_output = None
        if config.electra_objective:
            disc_output = self._get_discriminator_output(
                fake_data.inputs, self.discriminator, fake_data.is_fake_tokens,
                is_training=is_training)
            total_loss += config.disc_weight * disc_output.loss

        # Evaluation inputs
        eval_fn_inputs = {
            "input_ids": masked_inputs.input_ids,
            "masked_lm_preds": mlm_output.preds,
            "mlm_loss": mlm_output.per_example_loss,
            "masked_lm_ids": masked_inputs.masked_lm_ids,
            "masked_lm_weights": masked_inputs.masked_lm_weights,
            "input_mask": masked_inputs.input_mask
        }
        if config.electra_objective:
            eval_fn_inputs.update({
                "disc_loss": disc_output.per_example_loss,
                "disc_labels": disc_output.labels,
                "disc_probs": disc_output.probs,
                "disc_preds": disc_output.preds,
                "sampled_tokids": tf.argmax(fake_data.sampled_tokens, -1,
                                            output_type=tf.int32)
            })

        return total_loss, eval_fn_inputs

    def _get_masked_lm_output(self, inputs, generator, is_training=False):
        """Masked language modeling softmax layer."""
        masked_lm_weights = inputs.masked_lm_weights

        if self._config.uniform_generator:
            logits = tf.zeros(self.disc_config.vocab_size)
            logits_tiled = tf.zeros(
                pretrain_utils.get_shape_list(inputs.masked_lm_ids) +
                [self.disc_config.vocab_size])
            logits_tiled += tf.reshape(logits, [1, 1, self.disc_config.vocab_size])
            logits = logits_tiled
        else:
            outputs = generator(
                input_ids=inputs.input_ids,
                attention_mask=inputs.input_mask,
                token_type_ids=inputs.segment_ids,
                training=is_training)
            logits = outputs[0]
            logits = pretrain_utils.gather_positions(
                logits, inputs.masked_lm_positions)

        oh_labels = tf.one_hot(
            inputs.masked_lm_ids, depth=self.disc_config.vocab_size,
            dtype=tf.float32)

        probs = tf.cast(tf.nn.softmax(logits), tf.float32)
        log_probs = tf.cast(tf.nn.log_softmax(logits), tf.float32)
        label_log_probs = -tf.reduce_sum(log_probs * oh_labels, axis=-1)

        numerator = tf.reduce_sum(masked_lm_weights * label_log_probs)
        denominator = tf.reduce_sum(masked_lm_weights) + 1e-6
        loss = numerator / denominator
        preds = tf.argmax(log_probs, axis=-1, output_type=tf.int32)

        MLMOutput = collections.namedtuple(
            "MLMOutput", ["logits", "probs", "loss", "per_example_loss", "preds"])
        return MLMOutput(
            logits=logits, probs=probs, per_example_loss=label_log_probs,
            loss=loss, preds=preds)

    def _get_discriminator_output(self, inputs, discriminator, labels, is_training=False):
        """Discriminator binary classifier."""

        outputs = discriminator(
            input_ids=inputs.input_ids,
            attention_mask=inputs.input_mask,
            token_type_ids=inputs.segment_ids,
            training=is_training,
        )
        logits = outputs[0]
        weights = tf.cast(inputs.input_mask, tf.float32)
        labelsf = tf.cast(labels, tf.float32)
        logits = tf.cast(logits, tf.float32)
        losses = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=labelsf) * weights
        per_example_loss = (tf.reduce_sum(losses, axis=-1) /
                            (1e-6 + tf.reduce_sum(weights, axis=-1)))
        loss = tf.reduce_sum(losses) / (1e-6 + tf.reduce_sum(weights))
        probs = tf.nn.sigmoid(logits)
        preds = tf.cast(tf.round((tf.sign(logits) + 1) / 2), tf.int32)
        DiscOutput = collections.namedtuple(
            "DiscOutput", ["loss", "per_example_loss", "probs", "preds",
                           "labels"])
        return DiscOutput(
            loss=loss, per_example_loss=per_example_loss, probs=probs,
            preds=preds, labels=labels,
        )

    def _get_fake_data(self, inputs, mlm_logits):
        """Sample from the generator to create corrupted input."""
        inputs = pretrain_utils.unmask(inputs)
        disallow = tf.one_hot(
            inputs.masked_lm_ids, depth=self.disc_config.vocab_size,
            dtype=tf.float32) if self._config.disallow_correct else None
        sampled_tokens = tf.stop_gradient(pretrain_utils.sample_from_softmax(
            mlm_logits / self._config.temperature, disallow=disallow))
        sampled_tokids = tf.argmax(sampled_tokens, -1, output_type=tf.int32)
        updated_input_ids, masked = pretrain_utils.scatter_update(
            inputs.input_ids, sampled_tokids, inputs.masked_lm_positions)
        labels = masked * (1 - tf.cast(
            tf.equal(updated_input_ids, inputs.input_ids), tf.int32))
        updated_inputs = pretrain_utils.get_updated_inputs(
            inputs, input_ids=updated_input_ids)
        FakedData = collections.namedtuple("FakedData", [
            "inputs", "is_fake_tokens", "sampled_tokens"])
        return FakedData(inputs=updated_inputs, is_fake_tokens=labels,
                         sampled_tokens=sampled_tokens)


@add_start_docstrings(
    """
Electra model with a token classification head on top.

Both the discriminator and generator may be loaded into this model.""",
    ELECTRA_START_DOCSTRING,
)
class TFElectraForTokenClassification(TFElectraPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.electra = TFElectraMainLayer(config, name="electra")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier")

    @add_start_docstrings_to_callable(ELECTRA_INPUTS_DOCSTRING)
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        training=False,
    ):
        r"""
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.ElectraConfig`) and inputs:
        scores (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when :obj:`config.output_hidden_states=True`):
            tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``config.output_attentions=True``):
            tuple of :obj:`tf.Tensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`:

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import tensorflow as tf
        from transformers import ElectraTokenizer, TFElectraForTokenClassification

        tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
        model = TFElectraForTokenClassification.from_pretrained('google/electra-small-discriminator')
        input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
        outputs = model(input_ids)
        scores = outputs[0]
        """

        discriminator_hidden_states = self.electra(
            input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, training=training
        )
        discriminator_sequence_output = discriminator_hidden_states[0]
        discriminator_sequence_output = self.dropout(discriminator_sequence_output)
        logits = self.classifier(discriminator_sequence_output)
        output = (logits,)
        output += discriminator_hidden_states[1:]

        return output  # (loss), scores, (hidden_states), (attentions)


