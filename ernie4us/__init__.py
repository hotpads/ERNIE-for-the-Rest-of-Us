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

# Note: we don't necessary have new model artifacts for every version
# as the newer version may only have slight API changes
DEFAULT_MODEL_BINARY_REPO_URL_ = 'https://github.com/hotpads/ERNIE-for-the-Rest-of-Us/releases/download/0.1.0/'

# model variants
ERNIE_BASE_EN = 'ERNIE_Base_en_stable-2.0.0'
ERNIE_LARGE_EN = 'ERNIE_Large_en_stable-2.0.0'


def _dataclass_top_level_as_tuple(dataclass_instance):
    """
    `dataclasses.astuple()` does recursive traversal and would cause error converting fields of type tf.Tensor.
    This method converts a dataclass instance to tuple by `getattr` over the fields.
    """
    import dataclasses
    return tuple(getattr(dataclass_instance, field.name) for field in dataclasses.fields(dataclass_instance))


@dataclass
class Ernie2Input:
    token_ids: Union[tf.Tensor, np.array]
    sentence_ids: Union[tf.Tensor, np.array]
    task_ids: Union[tf.Tensor, np.array]
    input_mask: Union[tf.Tensor, np.array]

    def as_tuple(self):
        return _dataclass_top_level_as_tuple(self)


@dataclass
class Ernie2Output:
    sequence_features: Union[tf.Tensor, np.array]
    classification_features: Union[tf.Tensor, np.array]

    def as_tuple(self):
        return _dataclass_top_level_as_tuple(self)


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
        sentence_ids = self.pad_input_seq(sentence_ids, self.pad_id, dtype=np.int64)
        input_mask = self.pad_input_seq([1] * len(tokens), 0, np.float32)
        task_ids = np.full_like(token_ids, task_id)

        token_ids = np.reshape(token_ids, (1, -1))
        sentence_ids = np.reshape(sentence_ids, (1, -1))
        input_mask = np.reshape(input_mask, (1, -1))
        task_ids = np.reshape(task_ids, (1, -1))
        return Ernie2Input(token_ids, sentence_ids, task_ids, input_mask)


def create_ernie_model(model_name,
                       ernie_config_path,
                       ernie_vocab_path,
                       ernie_param_path,
                       max_seq_len,
                       do_lower_case=True):
    ernie_config = modeling.ErnieConfig.from_json_file(ernie_config_path)
    if model_name == ERNIE_LARGE_EN:
        ernie_config.intermediate_size = 4096
    with open(ernie_param_path, 'rb') as f:
        ernie_params = pickle.load(f)
    model = modeling.ErnieModel(
        max_seq_length=max_seq_len,
        config=ernie_config,
        ernie_params=ernie_params,
        use_one_hot_embeddings=False)
    for key in ernie_params.keys():
        _logger.error(f"found unexpected left-over ernie parameter {key}")
    assert len(ernie_params) == 0
    input_builder = Ernie2InputBuilder(ernie_vocab_path, do_lower_case=do_lower_case, max_seq_len=max_seq_len)
    return input_builder, model


def load_ernie_model(model_name, model_path, max_seq_len=200, do_lower_case=True,
                     model_binary_repository_url=None):
    ernie_tf_checkpoint_path, config_path, vocab_path, param_path = download_model_files(model_name, model_path,
        model_binary_repository_url=model_binary_repository_url)
    return create_ernie_model(model_name, config_path, vocab_path, param_path,
                              max_seq_len=max_seq_len, do_lower_case=do_lower_case)