#   Copyright (c) 2020 Zillow Groups. All Rights Reserved.
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

from typing import Union
from dataclasses import dataclass
import numpy as np
import os
import shutil
import pickle
from urllib.request import urlopen
import logging
import tensorflow as tf
from . import tokenization
from . import modeling


_logger = logging.getLogger(__name__)

ERNIE4US_VERSION = '0.1.0'

# model variants
ERNIE_BASE_EN = 'ERNIE_Base_en_stable-2.0.0'
ERNIE_LARGE_EN = 'ERNIE_Large_en_stable-2.0.0'


@dataclass
class Ernie2Input:
    token_ids: Union[tf.Tensor, np.array]
    sentence_ids: Union[tf.Tensor, np.array]
    position_ids: Union[tf.Tensor, np.array]
    task_ids: Union[tf.Tensor, np.array]
    input_mask: Union[tf.Tensor, np.array]


@dataclass
class Ernie2Output:
    sequence_features: Union[tf.Tensor, np.array]
    classification_features: Union[tf.Tensor, np.array]


DEFAULT_MODEL_BINARY_REPO_URL_ = 'https://github.com/hotpads/ERNIE-for-the-Rest-of-Us/releases/download/%s/' \
                                 % ERNIE4US_VERSION


def _download_file(url, destination_path):
    if not os.path.exists(destination_path):
        _logger.info('Downloading model artifact %s from %s' % (os.path.basename(url), url))
        with urlopen(url) as content:
            temp_path = destination_path + '.tmp'
            try:
                os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                with open(temp_path, 'wb') as out:
                    total = 0
                    while True:
                        stuff = content.read(4096)
                        total += len(stuff)
                        if len(stuff) == 0:
                            break
                        out.write(stuff)
                shutil.move(temp_path, destination_path)
            except Exception as e:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise e
            _logger.info('Downloaded model artifact (%d bytes) to %s' % (total, destination_path))


def download_model_files(model_name, path, model_binary_repository_url=None):
    path = os.path.join(path, model_name)
    if model_binary_repository_url is None:
        model_binary_repository_url = DEFAULT_MODEL_BINARY_REPO_URL_
    param_path = os.path.join(path, "%s_persistables.pkl" % model_name)
    vocab_path = os.path.join(path, "%s_vocab.txt" % model_name)
    config_path = os.path.join(path, "%s_config.json" % model_name)
    for file_path in [config_path, param_path, vocab_path]:
        _download_file(model_binary_repository_url + os.path.basename(file_path), file_path)
    return path, config_path, vocab_path, param_path


def get_is_training_tensors(is_training_tensor_name='is_training'):
    with tf.name_scope(""):
        try:
            tf_is_training = tf.get_default_graph().get_tensor_by_name('%s:0' % is_training_tensor_name)
            tf_is_training_float = tf.get_default_graph().get_tensor_by_name('%s_float:0' % is_training_tensor_name)
            return tf_is_training, tf_is_training_float
        except KeyError:
            tf_is_training = tf.placeholder_with_default(np.array(False), tuple(), name=is_training_tensor_name)
            tf_is_training_float = tf.cast(tf_is_training, tf.float32, name='%s_float' % is_training_tensor_name)
            return tf_is_training, tf_is_training_float


def get_dropout_rate_tensor(dropout_rate, is_training_tensor_name='is_training'):
    _, tf_is_training_float = get_is_training_tensors(is_training_tensor_name)
    return tf.multiply(tf_is_training_float, dropout_rate)


class Ernie2InputBuilder:
    def __init__(self, vocaburary_path, do_lower_case=True, max_seq_len=200):
        self._max_seq_len = max_seq_len
        self.tokenizer = tokenization.FullTokenizer(vocaburary_path, do_lower_case=do_lower_case)
        vocab = self.tokenizer.vocab
        self.pad_id = vocab["[PAD]"]

    @property
    def max_seq_len(self):
        return self._max_seq_len

    def pad_input_seq(self, input_seq, pad_value, dtype=np.int64):
        padded = input_seq + [pad_value] * (self.max_seq_len - len(input_seq))
        return np.array(padded, dtype=dtype)

    def truncate_sentences(self, sentences):
        """Truncates token sequences of sentences in place to the maximum length.
        Param:
            sentences: array of token arrays
        """
        # credit to ERNIE codes
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        max_length = self.max_seq_len - 1 - len(sentences)
        while True:
            total_length = 0
            max_tokens = 0
            longest = None
            for sentence in sentences:
                l = len(sentence)
                total_length += l
                if l > max_tokens:
                    max_tokens = l
                    longest = sentence
            if total_length <= max_length:
                break
            if longest:
                longest.pop()

    def build(self, text_a, text_b=None, task_id=0):
        sentences = []
        assert text_a is not None
        for sentence in [text_a, text_b]:
            if sentence is None:
                continue
            sentence = tokenization.convert_to_unicode(sentence)
            sentence = self.tokenizer.tokenize(sentence)
            sentences.append(sentence)
        self.truncate_sentences(sentences)
        tokens = ['[CLS]']
        sentence_ids = [0]
        for i, sentence_tokens in enumerate(sentences):
            tokens.extend(sentence_tokens)
            sentence_ids.extend([i] * len(sentence_tokens))
            tokens.append('[SEP]')
            sentence_ids.append(i)

        token_ids = self.pad_input_seq(self.tokenizer.convert_tokens_to_ids(tokens), self.pad_id, dtype=np.int64)
        position_ids = self.pad_input_seq(list(range(len(tokens))), self.pad_id, dtype=np.int64)
        sentence_ids = self.pad_input_seq(sentence_ids, self.pad_id, dtype=np.int64)
        input_mask = self.pad_input_seq([1] * len(tokens), 0, np.float32)
        task_ids = np.full_like(token_ids, task_id)

        token_ids = np.reshape(token_ids, (1, -1))
        sentence_ids = np.reshape(sentence_ids, (1, -1))
        position_ids = np.reshape(position_ids, (1, -1))
        input_mask = np.reshape(input_mask, (1, -1))
        task_ids = np.reshape(task_ids, (1, -1))
        return Ernie2Input(token_ids, sentence_ids, position_ids, task_ids, input_mask)


def create_ernie_model(model_name,
                       ernie_config_path,
                       ernie_vocab_path,
                       ernie_param_path,
                       max_seq_len, do_lower_case=True):
    ernie_config = modeling.ErnieConfig.from_json_file(ernie_config_path)
    if model_name == ERNIE_LARGE_EN:
        ernie_config.intermediate_size = 4096
    src_ids = tf.placeholder(tf.int32, (None, max_seq_len), name='src_ids')
    segment_ids = tf.placeholder(tf.int32, (None, max_seq_len), name='sent_ids')
    input_mask = tf.placeholder(tf.int32, (None, max_seq_len), name='input_mask')
    task_ids = tf.placeholder(tf.int32, (None, max_seq_len), name='task_ids')
    # these two are not used by the original BERT code, but have it to be compatible with ERNIE inputs
    pos_ids = tf.placeholder(tf.int32, (None, max_seq_len), name='pos_ids')

    with open(ernie_param_path, 'rb') as f:
        ernie_params = pickle.load(f)
    model = modeling.ErnieModel(
        config=ernie_config,
        input_ids=src_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        task_type_ids=task_ids,
        ernie_params=ernie_params,
        use_one_hot_embeddings=False)
    assert len(ernie_params) == 0
    ernie_tf_inputs = Ernie2Input(src_ids, segment_ids, pos_ids, task_ids, input_mask)
    ernie_tf_outputs = Ernie2Output(model.get_sequence_output(), model.get_pooled_output())
    input_builder = Ernie2InputBuilder(ernie_vocab_path, do_lower_case=do_lower_case, max_seq_len=max_seq_len)

    return input_builder, ernie_tf_inputs, ernie_tf_outputs


def load_ernie_model(model_name, model_path, max_seq_len=512, do_lower_case=True,
                     model_binary_repository_url=None):
    ernie_tf_checkpoint_path, config_path, vocab_path, param_path = download_model_files(model_name, model_path,
        model_binary_repository_url=model_binary_repository_url)
    return create_ernie_model(model_name, config_path, vocab_path, param_path,
                              max_seq_len=max_seq_len, do_lower_case=do_lower_case)