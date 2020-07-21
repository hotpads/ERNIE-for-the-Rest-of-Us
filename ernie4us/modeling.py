# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""ERNIE inference model. As ERNIE uses the same Transformer architecture, this code is largely adopted from
[BERT](https://github.com/google-research/bert)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict
import copy
import json
import math
import re
import six
import numpy as np
import tensorflow as tf


class ErnieConfig(object):
  """Configuration for `ErnieModel`."""

  def __init__(self,
               vocab_size,
               hidden_size=768,
               num_hidden_layers=12,
               num_attention_heads=12,
               intermediate_size=3072,
               hidden_act="gelu",
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               max_position_embeddings=512,
               sent_type_vocab_size=4,
               task_type_vocab_size=16,
               initializer_range=0.02):
    """Constructs ErnieConfig.

    Args:
      vocab_size: Vocabulary size of `inputs_ids` in `ErnieModel`.
      hidden_size: Size of the encoder layers and the pooler layer.
      num_hidden_layers: Number of hidden layers in the Transformer encoder.
      num_attention_heads: Number of attention heads for each attention layer in
        the Transformer encoder.
      intermediate_size: The size of the "intermediate" (i.e., feed-forward)
        layer in the Transformer encoder.
      hidden_act: The non-linear activation function (function or string) in the
        encoder and pooler.
      hidden_dropout_prob: The dropout probability for all fully connected
        layers in the embeddings, encoder, and pooler.
      attention_probs_dropout_prob: The dropout ratio for the attention
        probabilities.
      max_position_embeddings: The maximum sequence length that this model might
        ever be used with. Typically set this to something large just in case
        (e.g., 512 or 1024 or 2048).
      sent_type_vocab_size: The vocabulary size of the `token_type_ids` passed into
        `ErnieModel`. This is the same as the `type_vocab_size` in the original BERT model.
      task_type_vocab_size: The vocaburary size of the `task_type_ids`.
      initializer_range: The stdev of the truncated_normal_initializer for
        initializing all weight matrices.
    """
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.sent_type_vocab_size = sent_type_vocab_size
    self.task_type_vocab_size = task_type_vocab_size
    self.initializer_range = initializer_range

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `ErnieConfig` from a Python dictionary of parameters."""
    config = ErnieConfig(vocab_size=None)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config

  @classmethod
  def from_json_file(cls, json_file):
    """Constructs a `ErnieConfig` from a json file of parameters."""
    with open(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class ErnieModel(tf.keras.layers.Layer):
  """
  Example usage:

  ```python
  # Already been converted into WordPiece token ids
  input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
  input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
  token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

  config = modeling.ErnieConfig(vocab_size=32000, hidden_size=512,
    num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

  model = modeling.ErnieModel(config=config, is_training=True,
    input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

  label_embeddings = tf.get_variable(...)
  pooled_output = model.get_pooled_output()
  logits = tf.matmul(pooled_output, label_embeddings)
  ...
  ```
  """

  def __init__(self,
               max_seq_length: int,
               config: ErnieConfig,
               ernie_params: Dict[str, np.ndarray],
               use_one_hot_embeddings=False,
               scope='ernie',
               *args, **kwargs):
    """Constructor for ErnieModel. Please note that unlike the original BERT codes, the `is_training` argument is
    removed and the dropout is controlled by a tensorflow placeholder named `is_training` of type `bool` with default
    value `False`.

    Args:
      config: `ErnieConfig` instance.
      is_training: bool. rue for training model, false for eval model. Controls
        whether dropout will be applied.
      input_ids: int32 Tensor of shape [batch_size, seq_length].
      input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
      token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
        embeddings or tf.embedding_lookup() for the word embeddings. On the TPU,
        it is must faster if this is True, on the CPU or GPU, it is faster if
        this is False.
      scope: (optional) variable scope. Defaults to "ernie".

    :return
      A tuple of the last pooled output, sequence output

    Raises:
      ValueError: The config is invalid or one of the input tensor shapes
        is invalid.
    """
    super().__init__(*args, **kwargs)
    self._max_seq_length = max_seq_length
    self._ernie_config = copy.deepcopy(config)
    self._use_one_hot_embeddings = use_one_hot_embeddings
    self._root_scope = scope

    with tf.name_scope(self._root_scope):
      with tf.name_scope("embeddings"):
        self._pre_encoder_layer_norm = tf.keras.layers.LayerNormalization(
          axis=-1,
          epsilon=1e-12,
          beta_initializer=const_initializer(ernie_params, f"pre_encoder_layer_norm_bias"),
          gamma_initializer=const_initializer(ernie_params, f"pre_encoder_layer_norm_scale"),
          name='pre_encoder_layer_norm')

      with tf.name_scope("encoder"):
        self._encoder = TransformerLayer(
          ernie_params=ernie_params,
          hidden_size=self._ernie_config.hidden_size,
          num_hidden_layers=self._ernie_config.num_hidden_layers,
          num_attention_heads=self._ernie_config.num_attention_heads,
          intermediate_size=self._ernie_config.intermediate_size,
          intermediate_act_fn=get_activation(self._ernie_config.hidden_act),
          hidden_dropout_prob=self._ernie_config.hidden_dropout_prob,
          attention_probs_dropout_prob=self._ernie_config.attention_probs_dropout_prob,
          initializer_range=self._ernie_config.initializer_range,
          do_return_all_layers=True)

      with tf.name_scope("pooler"):
        self._pooled_fc = tf.keras.layers.Dense(
          self._ernie_config.hidden_size,
          name='classification_features',
          activation=tf.tanh,
          kernel_initializer=const_initializer(ernie_params, 'pooled_fc.w_0'),
          bias_initializer=const_initializer(ernie_params, 'pooled_fc.b_0'))

    assert len(ernie_params) > 0

    with tf.name_scope(self._root_scope):
      with tf.name_scope("embeddings"):
        ernie_param = ernie_params.pop('word_embedding')
        self._embedding_table = self.add_weight(
          name="word_embeddings",
          shape=ernie_param.shape,
          initializer=tf.constant_initializer(ernie_param))

        ernie_param = ernie_params.pop('sent_embedding')
        self._token_type_table = self.add_weight(
          name='token_type_embeddings',
          shape=ernie_param.shape,
          initializer=tf.constant_initializer(ernie_param))

        ernie_param = ernie_params.pop('pos_embedding')
        self._full_position_embeddings = self.add_weight(
          name='position_embeddings',
          shape=ernie_param.shape,
          initializer=tf.constant_initializer(ernie_param))

        ernie_param = ernie_params.pop('task_embedding')
        self._task_type_table = self.add_weight(
          name='task_embeddings',
          shape=ernie_param.shape,
          initializer=tf.constant_initializer(ernie_param))

  @tf.function
  def call(self, inputs, training=False, **kwargs):
    if isinstance(inputs, (list, tuple)):
      input_ids, token_type_ids, task_type_ids, input_token_mask = inputs
    else:
      input_ids = inputs
      input_token_mask, token_type_ids, task_type_ids = None, None, None

    batch_size, seq_length = get_shape_list(input_ids, expected_rank=2)
    if input_token_mask is None:
      input_token_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

    if token_type_ids is None:
      token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

    if task_type_ids is None:
      task_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

    with tf.name_scope(self._root_scope):
      with tf.name_scope("embeddings"):
        # Perform embedding lookup on the word ids.
        embedding_output = embedding_lookup(
            embedding_table=self._embedding_table,
            input_ids=input_ids,
            vocab_size=self._ernie_config.vocab_size,
            embedding_size=self._ernie_config.hidden_size,
            use_one_hot_embeddings=self._use_one_hot_embeddings)

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        embedding_output = embedding_postprocessor(
            token_type_table=self._token_type_table,
            full_position_embeddings=self._full_position_embeddings,
            task_type_table=self._task_type_table,
            input_tensor=embedding_output,
            training=training,
          pre_encoder_layer_norm=self._pre_encoder_layer_norm,
            token_type_ids=token_type_ids,
            task_type_ids=task_type_ids,
            max_position_embeddings=self._ernie_config.max_position_embeddings,
            dropout_prob=self._ernie_config.hidden_dropout_prob)

      with tf.name_scope("encoder"):
        # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
        # mask of shape [batch_size, seq_length, seq_length] which is used
        # for the attention scores.
        attention_mask = create_attention_mask_from_input_mask(input_ids, input_token_mask)

        # Run the stacked transformer.
        # `sequence_output` shape = [batch_size, seq_length, hidden_size].
        all_encoder_layers = self._encoder([embedding_output, attention_mask], training=training)

      sequence_output = tf.identity(all_encoder_layers[-1], 'sequence_features')
      # The "pooler" converts the encoded sequence tensor of shape
      # [batch_size, seq_length, hidden_size] to a tensor of shape
      # [batch_size, hidden_size]. This is necessary for segment-level
      # (or segment-pair-level) classification tasks where we need a fixed
      # dimensional representation of the segment.
      with tf.name_scope("pooler"):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We assume that this has been pre-trained
        first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
        pooled_output = self._pooled_fc(first_token_tensor)
    return sequence_output, pooled_output

  @property
  def embedding_table(self):
    return self._embedding_table

  @property
  def max_seq_length(self):
    return self._max_seq_length

  def create_keras_invocation(self):
    """
    Convenient method to create `tf.keras.Input` and apply to the layer. Same as
    ```python
    ernie_layer = ErnieModel(...)
    src_ids = tf.keras.Input((None, self.max_seq_length), dtype=tf.int32, name='src_ids')
    ... other inputs ...
    pooled_output, sequence_output, embedding_output = ernie_layer([src_ids, ...])
    ```
    :return: a tuple of the followings:
      * all keras `Input` wrapped in an `Ernie2Input` instance
      * pooled_output and sequence_output wrapped in an `Ernie2Output` instance
    """
    from . import Ernie2Input, Ernie2Output
    src_ids = tf.keras.Input((self._max_seq_length,), dtype=tf.int32, name='src_ids')
    segment_ids = tf.keras.Input((self._max_seq_length,), dtype=tf.int32, name='segment_ids')
    input_mask = tf.keras.Input((self._max_seq_length,), dtype=tf.int32, name='input_mask')
    task_ids = tf.keras.Input((self._max_seq_length,), dtype=tf.int32, name='task_ids')
    ernie_tf_inputs = Ernie2Input(src_ids, segment_ids, task_ids, input_mask)
    sequence_output, pooled_output = self(ernie_tf_inputs.as_tuple())
    ernie_tf_outputs = Ernie2Output(sequence_features=sequence_output, classification_features=pooled_output)
    return ernie_tf_inputs, ernie_tf_outputs


def gelu(input_tensor):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415

  Args:
    input_tensor: float Tensor to perform activation.

  Returns:
    `input_tensor` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.math.erf(input_tensor / tf.sqrt(2.0)))
  return input_tensor * cdf


def get_activation(activation_string):
  """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

  Args:
    activation_string: String name of the activation function.

  Returns:
    A Python function corresponding to the activation function. If
    `activation_string` is None, empty, or "linear", this will return None.
    If `activation_string` is not a string, it will return `activation_string`.

  Raises:
    ValueError: The `activation_string` does not correspond to a known
      activation.
  """

  # We assume that anything that"s not a string is already an activation
  # function, so we just return it.
  if not isinstance(activation_string, six.string_types):
    return activation_string

  if not activation_string:
    return None

  act = activation_string.lower()
  if act == "linear":
    return None
  elif act == "relu":
    return tf.nn.relu
  elif act == "gelu":
    return gelu
  elif act == "tanh":
    return tf.tanh
  else:
    raise ValueError("Unsupported activation: %s" % act)


def _dropout(input_tensor, dropout_prob, training):
  if training:
    return tf.nn.dropout(input_tensor, rate=dropout_prob)
  else:
    return input_tensor


def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


def embedding_lookup(embedding_table,
                     input_ids,
                     vocab_size,
                     embedding_size=128,
                     use_one_hot_embeddings=False):
  """Looks up words embeddings for id tensor.

  Args:
    input_ids: int32 Tensor of shape [batch_size, seq_length] containing word
      ids.
    vocab_size: int. Size of the embedding vocabulary.
    embedding_size: int. Width of the word embeddings.
    use_one_hot_embeddings: bool. If True, use one-hot method for word
      embeddings. If False, use `tf.nn.embedding_lookup()`. One hot is better
      for TPUs.

  Returns:
    float Tensor of shape [batch_size, seq_length, embedding_size].
  """
  # This function assumes that the input is of shape [batch_size, seq_length,
  # num_inputs].
  #
  # If the input is a 2D tensor of shape [batch_size, seq_length], we
  # reshape to [batch_size, seq_length, 1].
  if input_ids.shape.ndims == 2:
    input_ids = tf.expand_dims(input_ids, axis=[-1])

  if use_one_hot_embeddings:
    flat_input_ids = tf.reshape(input_ids, [-1])
    one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
    output = tf.matmul(one_hot_input_ids, embedding_table)
  else:
    output = tf.nn.embedding_lookup(embedding_table, input_ids)

  input_shape = get_shape_list(input_ids)

  output = tf.reshape(output,
                      input_shape[0:-1] + [input_shape[-1] * embedding_size])
  return output


def embedding_postprocessor(token_type_table: tf.Tensor,
                            full_position_embeddings: tf.Tensor,
                            task_type_table: tf.Tensor,
                            input_tensor: tf.Tensor,
                            pre_encoder_layer_norm: tf.keras.layers.LayerNormalization,
                            training,
                            token_type_ids=None,
                            task_type_ids=None,
                            max_position_embeddings=512,
                            dropout_prob=0.1):
  """Performs various post-processing on a word embedding tensor.

  Args:
    token_type_table: token type embeddings
    full_position_embeddings: position embeddings
    task_type_table: task type embeddings
    input_tensor: float Tensor of shape [batch_size, seq_length,
      embedding_size].
    token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      Must be specified if `use_token_type` is True.
    max_position_embeddings: int. Maximum sequence length that might ever be
      used with this model. This can be longer than the sequence length of
      input_tensor, but cannot be shorter.
    dropout_prob: float. Dropout probability applied to the final output tensor.

  Returns:
    float tensor with same shape as `input_tensor`.

  Raises:
    ValueError: One of the tensor shapes or input values is invalid.
  """
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  width = input_shape[2]

  if seq_length > max_position_embeddings:
    raise ValueError("The seq length (%d) cannot be greater than "
                     "`max_position_embeddings` (%d)" %
                     (seq_length, max_position_embeddings))

  output = input_tensor

  if token_type_table is not None:
    if token_type_ids is None:
      raise ValueError("`token_type_ids` must be specified if"
                       "`use_token_type` is True.")
    # This vocab will be small so we always do one-hot here, since it is always
    # faster for a small vocabulary.
    flat_token_type_ids = tf.reshape(token_type_ids, [-1])
    one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_table.shape[0])
    token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
    token_type_embeddings = tf.reshape(token_type_embeddings,
                                       [batch_size, seq_length, width])
    output += token_type_embeddings

  if full_position_embeddings is not None:
    # Since the position embedding table is a learned variable, we create it
    # using a (long) sequence length `max_position_embeddings`. The actual
    # sequence length might be shorter than this, for faster training of
    # tasks that do not have long sequences.
    #
    # So `full_position_embeddings` is effectively an embedding table
    # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
    # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
    # perform a slice.
    if seq_length < max_position_embeddings:
      position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                     [seq_length, -1])
    else:
      position_embeddings = full_position_embeddings

    num_dims = len(output.shape.as_list())

    # Only the last two dimensions are relevant (`seq_length` and `width`), so
    # we broadcast among the first dimensions, which is typically just
    # the batch size.
    position_broadcast_shape = []
    for _ in range(num_dims - 2):
      position_broadcast_shape.append(1)
    position_broadcast_shape.extend([seq_length, width])
    position_embeddings = tf.reshape(position_embeddings,
                                     position_broadcast_shape)
    output += position_embeddings

  # This vocab will be small so we always do one-hot here, since it is always
  # faster for a small vocabulary.
  flat_task_type_ids = tf.reshape(task_type_ids, [-1])
  one_hot_ids = tf.one_hot(flat_task_type_ids, depth=task_type_table.shape[0])
  task_type_embeddings = tf.matmul(one_hot_ids, task_type_table)
  task_type_embeddings = tf.reshape(task_type_embeddings,
                                     [batch_size, seq_length, width])
  output += task_type_embeddings
  output = pre_encoder_layer_norm(output)
  output = _dropout(output, dropout_prob, training)
  return output


def create_attention_mask_from_input_mask(from_tensor, to_mask):
  """Create 3D attention mask from a 2D tensor mask.

  Args:
    from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
    to_mask: int32 Tensor of shape [batch_size, to_seq_length].

  Returns:
    float Tensor of shape [batch_size, from_seq_length, to_seq_length].
  """
  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  batch_size = from_shape[0]
  from_seq_length = from_shape[1]

  to_shape = get_shape_list(to_mask, expected_rank=2)
  to_seq_length = to_shape[1]

  to_mask = tf.cast(
      tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

  # We don't assume that `from_tensor` is a mask (although it could be). We
  # don't actually care if we attend *from* padding tokens (only *to* padding)
  # tokens so we create a tensor of all ones.
  #
  # `broadcast_ones` = [batch_size, from_seq_length, 1]
  broadcast_ones = tf.ones(
      shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

  # Here we broadcast along two dimensions to create the mask.
  mask = broadcast_ones * to_mask

  return mask


class AttentionLayer(tf.keras.layers.Layer):

  def __init__(self,
               ernie_params: Dict[str, np.ndarray],
               num_attention_heads=1,
               size_per_head=512,
               query_act=None,
               key_act=None,
               value_act=None,
               attention_probs_dropout_prob=0.0,
               do_return_2d_tensor=False,
               **kwargs):
    """Performs multi-headed attention from `from_tensor` to `to_tensor`.

    This is an implementation of multi-headed attention based on "Attention
    is all you Need". If `from_tensor` and `to_tensor` are the same, then
    this is self-attention. Each timestep in `from_tensor` attends to the
    corresponding sequence in `to_tensor`, and returns a fixed-with vector.

    This function first projects `from_tensor` into a "query" tensor and
    `to_tensor` into "key" and "value" tensors. These are (effectively) a list
    of tensors of length `num_attention_heads`, where each tensor is of shape
    [batch_size, seq_length, size_per_head].

    Then, the query and key tensors are dot-producted and scaled. These are
    softmaxed to obtain attention probabilities. The value tensors are then
    interpolated by these probabilities, then concatenated back to a single
    tensor and returned.

    In practice, the multi-headed attention are done with transposes and
    reshapes rather than actual separate tensors.

    Args:
      from_tensor: float Tensor of shape [batch_size, from_seq_length,
        from_width].
      to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
      attention_mask: (optional) int32 Tensor of shape [batch_size,
        from_seq_length, to_seq_length]. The values should be 1 or 0. The
        attention scores will effectively be set to -infinity for any positions in
        the mask that are 0, and will be unchanged for positions that are 1.
      num_attention_heads: int. Number of attention heads.
      size_per_head: int. Size of each attention head.
      query_act: (optional) Activation function for the query transform.
      key_act: (optional) Activation function for the key transform.
      value_act: (optional) Activation function for the value transform.
      attention_probs_dropout_prob:
      initializer_range: float. Range of the weight initializer.
      do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
        * from_seq_length, num_attention_heads * size_per_head]. If False, the
        output will be of shape [batch_size, from_seq_length, num_attention_heads
        * size_per_head].

    Returns:
      float Tensor of shape [batch_size, from_seq_length,
        num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
        true, this will be of shape [batch_size * from_seq_length,
        num_attention_heads * size_per_head]).

    Raises:
      ValueError: Any of the arguments or tensor shapes are invalid.
    """
    super().__init__(**kwargs)
    self._do_return_2d_tensor = do_return_2d_tensor
    self._probs_dropout_prob = attention_probs_dropout_prob
    self._value_activation = value_act
    self._key_activation = key_act
    self._query_activation = query_act
    self._size_per_head = size_per_head
    self._num_attention_heads = num_attention_heads

    paddle_prefix = f"{get_paddle_layer_prefix()}_multi_head_att"

    # `query_layer` = [B*F, N*H]
    self._query_layer = tf.keras.layers.Dense(
      self._num_attention_heads * self._size_per_head,
      activation=self._query_activation,
      name="query",
      kernel_initializer=const_initializer(ernie_params, f"{paddle_prefix}_query_fc.w_0"),
      bias_initializer=const_initializer(ernie_params, f"{paddle_prefix}_query_fc.b_0"))

    # `key_layer` = [B*T, N*H]
    self._key_layer = tf.keras.layers.Dense(
      self._num_attention_heads * self._size_per_head,
      activation=self._key_activation,
      name="key",
      kernel_initializer=const_initializer(ernie_params, f"{paddle_prefix}_key_fc.w_0"),
      bias_initializer=const_initializer(ernie_params, f"{paddle_prefix}_key_fc.b_0"))

    # `value_layer` = [B*T, N*H]
    self._value_layer = tf.keras.layers.Dense(
      self._num_attention_heads * self._size_per_head,
      activation=self._value_activation,
      name="value",
      kernel_initializer=const_initializer(ernie_params, f"{paddle_prefix}_value_fc.w_0"),
      bias_initializer=const_initializer(ernie_params, f"{paddle_prefix}_value_fc.b_0"))

  def call(self, inputs, training=False, **kwargs):
    if self._do_return_2d_tensor:
      from_tensor, to_tensor, attention_mask, batch_size, to_seq_length, from_seq_length = inputs
    else:
      from_tensor, to_tensor, attention_mask = inputs
      from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
      to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

      tf.debugging.assert_equal(from_shape, to_shape, "The rank of `from_tensor` must match the rank of `to_tensor`.")
      batch_size = from_shape[0]
      from_seq_length = from_shape[1]
      to_seq_length = to_shape[1]
    return self.attention_layer(from_tensor, to_tensor, attention_mask,
                                batch_size, from_seq_length, to_seq_length,
                                training)

  def attention_layer(self, from_tensor, to_tensor, attention_mask,
                      batch_size, from_seq_length, to_seq_length,
                      training):
    def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                             seq_length, width):
      output_tensor = tf.reshape(
          input_tensor, [batch_size, seq_length, num_attention_heads, width])

      output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
      return output_tensor

    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`

    from_tensor_2d = reshape_to_matrix(from_tensor)
    to_tensor_2d = reshape_to_matrix(to_tensor)

    # `query_layer` = [B*F, N*H]
    query_layer = self._query_layer(from_tensor_2d)

    # `key_layer` = [B*T, N*H]
    key_layer = self._key_layer(to_tensor_2d)

    # `value_layer` = [B*T, N*H]
    value_layer = self._value_layer(to_tensor_2d)

    # `query_layer` = [B, N, F, H]
    query_layer = transpose_for_scores(query_layer, batch_size,
                                       self._num_attention_heads, from_seq_length,
                                       self._size_per_head)

    # `key_layer` = [B, N, T, H]
    key_layer = transpose_for_scores(key_layer, batch_size, self._num_attention_heads,
                                     to_seq_length, self._size_per_head)

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    # `attention_scores` = [B, N, F, T]
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(self._size_per_head)))

    if attention_mask is not None:
      # `attention_mask` = [B, 1, F, T]
      attention_mask = tf.expand_dims(attention_mask, axis=[1])

      # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
      # masked positions, this operation will create a tensor which is 0.0 for
      # positions we want to attend and -10000.0 for masked positions.
      adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

      # Since we are adding it to the raw scores before the softmax, this is
      # effectively the same as removing these entirely.
      attention_scores += adder

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, N, F, T]
    attention_probs = tf.nn.softmax(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = _dropout(attention_probs, self._probs_dropout_prob, training=training)

    # `value_layer` = [B, T, N, H]
    value_layer = tf.reshape(
        value_layer,
        [batch_size, to_seq_length, self._num_attention_heads, self._size_per_head])

    # `value_layer` = [B, N, T, H]
    value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

    # `context_layer` = [B, N, F, H]
    context_layer = tf.matmul(attention_probs, value_layer)

    # `context_layer` = [B, F, N, H]
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

    if self._do_return_2d_tensor:
      # `context_layer` = [B*F, N*V]
      context_layer = tf.reshape(
          context_layer,
          [batch_size * from_seq_length, self._num_attention_heads * self._size_per_head])
    else:
      # `context_layer` = [B, F, N*V]
      context_layer = tf.reshape(
          context_layer,
          [batch_size, from_seq_length, self._num_attention_heads * self._size_per_head])

    return context_layer


def get_current_scope():
  with tf.name_scope('_get_current_scope') as scope:
    return '/'.join(scope.split('/')).replace('_get_current_scope/', '')


def get_paddle_layer_prefix():
  tf_name_prefix = get_current_scope()
  paddle_layer = int(re.search(r'layer_(\d+)', tf_name_prefix).group(1))
  paddle_prefix = f"encoder_layer_{paddle_layer}"
  return paddle_prefix


def const_initializer(ernie_params: Dict[str, np.ndarray], key: str):
  return tf.constant_initializer(ernie_params.pop(key))


class TransformerLayer(tf.keras.layers.Layer):
  def __init__(self,
               ernie_params: Dict[str, np.ndarray],
               hidden_size=768,
               num_hidden_layers=12,
               num_attention_heads=12,
               intermediate_size=3072,
               intermediate_act_fn=gelu,
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               initializer_range=0.02,
               do_return_all_layers=False,
               **kwargs):
    """Multi-headed, multi-layer Transformer from "Attention is All You Need".

    This is almost an exact implementation of the original Transformer encoder.

    See the original paper:
    https://arxiv.org/abs/1706.03762

    Also see:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

    Args:
      input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
      attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
        seq_length], with 1 for positions that can be attended to and 0 in
        positions that should not be.
      hidden_size: int. Hidden size of the Transformer.
      training: boolean or tensor indicating training mode
      num_hidden_layers: int. Number of layers (blocks) in the Transformer.
      num_attention_heads: int. Number of attention heads in the Transformer.
      intermediate_size: int. The size of the "intermediate" (a.k.a., feed
        forward) layer.
      intermediate_act_fn: function. The non-linear activation function to apply
        to the output of the intermediate/feed-forward layer.
      hidden_dropout_prob: float. Dropout probability for the hidden layers.
      attention_probs_dropout_prob: float. Dropout probability of the attention
        probabilities.
      initializer_range: float. Range of the initializer (stddev of truncated
        normal).
      do_return_all_layers: Whether to also return all layers or just the final
        layer.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size], the final
      hidden layer of the Transformer.

    Raises:
      ValueError: A Tensor shape or parameter is invalid.
    """
    super().__init__(**kwargs)
    self._attention_probs_dropout_prob = attention_probs_dropout_prob
    self._hidden_size = hidden_size
    self._num_hidden_layers = num_hidden_layers
    self._num_attention_heads = num_attention_heads
    self._intermediate_size = intermediate_size
    self._intermediate_act_fn = intermediate_act_fn
    self._hidden_dropout_prob = hidden_dropout_prob
    self._do_return_all_layers = do_return_all_layers
    self._initializer_range = initializer_range
    if self._hidden_size % self._num_attention_heads != 0:
      raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (self._hidden_size, self._num_attention_heads))

    self._attention_layers = list()
    self._multi_head_att_output_fc = list()
    self._post_att_layer_norm = list()
    self._ffn_fc_0 = list()
    self._ffn_fc_1 = list()
    self._ffn_layer_norm = list()

    attention_head_size = int(self._hidden_size / self._num_attention_heads)

    for layer_idx in range(self._num_hidden_layers):
      with tf.name_scope("layer_%d" % layer_idx):
        with tf.name_scope("attention"):
          with tf.name_scope("self"):
            attention_layer = AttentionLayer(
              ernie_params=ernie_params,
              num_attention_heads=self._num_attention_heads,
              size_per_head=attention_head_size,
              attention_probs_dropout_prob=self._attention_probs_dropout_prob,
              do_return_2d_tensor=True)
            self._attention_layers.append(attention_layer)

          with tf.name_scope("output"):
            paddle_prefix = f"{get_paddle_layer_prefix()}_multi_head_att"
            attention_output = tf.keras.layers.Dense(
              self._hidden_size,
              activation=None,
              name="multi_head_att",
              kernel_initializer=const_initializer(ernie_params, f"{paddle_prefix}_output_fc.w_0"),
              bias_initializer=const_initializer(ernie_params, f"{paddle_prefix}_output_fc.b_0"))
            self._multi_head_att_output_fc.append(attention_output)

            paddle_prefix = f"{get_paddle_layer_prefix()}_post_att_layer_norm"
            layer_normalizer = tf.keras.layers.LayerNormalization(
              axis=-1,
              epsilon=1e-12,
              beta_initializer=const_initializer(ernie_params, f"{paddle_prefix}_bias"),
              gamma_initializer=const_initializer(ernie_params, f"{paddle_prefix}_scale"),
              name="post_att_layer_norm")
            self._post_att_layer_norm.append(layer_normalizer)

        # The activation is only applied to the "intermediate" hidden layer.
        with tf.name_scope("intermediate"):
          paddle_prefix = f"{get_paddle_layer_prefix()}_ffn_fc_0"
          intermediate_output = tf.keras.layers.Dense(
            self._intermediate_size,
            activation=self._intermediate_act_fn,
            name="ffn_fc_0",
            kernel_initializer=const_initializer(ernie_params, f"{paddle_prefix}.w_0"),
            bias_initializer=const_initializer(ernie_params, f"{paddle_prefix}.b_0"))
          self._ffn_fc_0.append(intermediate_output)

        # Down-project back to `hidden_size` then add the residual.
        with tf.name_scope("output"):
          paddle_prefix = f"{get_paddle_layer_prefix()}_ffn_fc_1"
          layer_output = tf.keras.layers.Dense(
            self._hidden_size,
            activation=None,
            name=f"ffn_fc_1",
            kernel_initializer=const_initializer(ernie_params, f"{paddle_prefix}.w_0"),
            bias_initializer=const_initializer(ernie_params, f"{paddle_prefix}.b_0"))
          self._ffn_fc_1.append(layer_output)

          paddle_prefix = f"{get_paddle_layer_prefix()}_post_ffn_layer_norm"
          layer_normalizer = tf.keras.layers.LayerNormalization(
            axis=-1,
            epsilon=1e-12,
            beta_initializer=const_initializer(ernie_params, f"{paddle_prefix}_bias"),
            gamma_initializer=const_initializer(ernie_params, f"{paddle_prefix}_scale"),
            name="post_ffn_layer_norm")
          self._ffn_layer_norm.append(layer_normalizer)

  def build(self, input_shape):
    input_tensor, _ = input_shape
    input_width = input_tensor[2]
    # The Transformer performs sum residuals on all layers so the input needs
    # to be the same as the hidden size.
    if input_width != self._hidden_size:
      raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                       (input_width, self._hidden_size))

  def call(self, inputs, training=False, **kwargs):
    input_tensor, attention_mask = inputs
    return self.transformer_model(input_tensor, attention_mask, training)

  def transformer_model(self, input_tensor, attention_mask, training):
    input_shape = get_shape_list(input_tensor, expected_rank=3)

    # We keep the representation as a 2D tensor to avoid re-shaping it back and
    # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
    # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
    # help the optimizer.

    input_tensor_shape = tf.shape(input_tensor)
    batch_size=input_tensor_shape[0]
    from_seq_length=input_tensor_shape[1]
    to_seq_length=input_tensor_shape[1]
    prev_output = reshape_to_matrix(input_tensor)

    all_layer_outputs = []
    for layer_idx in range(self._num_hidden_layers):
      with tf.name_scope("layer_%d" % layer_idx):
        layer_input = prev_output

        with tf.name_scope("attention"):
          attention_heads = []
          with tf.name_scope("self"):
            attention_head = self._attention_layers[layer_idx]([layer_input, layer_input, attention_mask,
                                                    batch_size, from_seq_length, to_seq_length],
                training=training)
            attention_heads.append(attention_head)

          if len(attention_heads) == 1:
            attention_output = attention_heads[0]
          else:
            # In the case where we have other sequences, we just concatenate
            # them to the self-attention head before the projection.
            attention_output = tf.concat(attention_heads, axis=-1)

          # Run a linear projection of `hidden_size` then add a residual
          # with `layer_input`.
          with tf.name_scope("output"):
            attention_output = self._multi_head_att_output_fc[layer_idx](attention_output)
            attention_output = _dropout(attention_output, self._hidden_dropout_prob, training=training)
            attention_output = self._post_att_layer_norm[layer_idx](attention_output + layer_input)

        # The activation is only applied to the "intermediate" hidden layer.
        with tf.name_scope("intermediate"):
          intermediate_output = self._ffn_fc_0[layer_idx](attention_output)

        # Down-project back to `hidden_size` then add the residual.
        with tf.name_scope("output"):
          layer_output = self._ffn_fc_1[layer_idx](intermediate_output)
          layer_output = _dropout(layer_output, self._hidden_dropout_prob, training=training)
          layer_output = self._ffn_layer_norm[layer_idx](layer_output + attention_output)
          prev_output = layer_output
          all_layer_outputs.append(layer_output)

    if self._do_return_all_layers:
      final_outputs = []
      for layer_output in all_layer_outputs:
        final_output = reshape_from_matrix(layer_output, input_shape)
        final_outputs.append(final_output)
      return final_outputs
    else:
      final_output = reshape_from_matrix(prev_output, input_shape)
      return final_output


def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape


def reshape_to_matrix(input_tensor):
  """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
  ndims = input_tensor.shape.ndims
  if ndims < 2:
    raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                     (input_tensor.shape))
  if ndims == 2:
    return input_tensor

  width = input_tensor.shape[-1]
  output_tensor = tf.reshape(input_tensor, [-1, width])
  return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
  """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
  if len(orig_shape_list) == 2:
    return output_tensor

  output_shape = get_shape_list(output_tensor)

  orig_dims = orig_shape_list[0:-1]
  width = output_shape[-1]

  return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = get_current_scope()
    raise ValueError(
        "For the tensor `%s` in scope `%s`, the actual rank "
        "`%d` (shape = %s) is not equal to the expected rank `%s`" %
        (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))
