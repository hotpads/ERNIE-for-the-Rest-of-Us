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

import os, sys
import pickle

# NOTE(paddle-dev): All of these flags should be
# set before `import paddle`. Otherwise, it would
# not take any effect.
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'  # enable gc

import paddle.fluid as fluid

try:
    from model.ernie import ErnieConfig
except ModuleNotFoundError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                    'third_party', 'ERNIE'))

from model.ernie import ErnieConfig
from utils.init import init_pretraining_params
from model.ernie import ErnieModel
from paddle.fluid.executor import _fetch_var as fetch_var


def create_model(args, ernie_config):
    input_names = ("src_ids", "sent_ids", "pos_ids", "task_ids", "input_mask")
    shapes=[[-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
            [-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
            [-1, args.max_seq_len, 1]]
    dtypes=[
        'int64', 'int64', 'int64', 'int64', 'float32'
    ]

    inputs = [fluid.data(name, shape, dtype=dtype) for name, shape, dtype in zip(input_names, shapes, dtypes)]
    (src_ids, sent_ids, pos_ids, task_ids, input_mask) = inputs

    ernie = ErnieModel(
        src_ids=src_ids,
        position_ids=pos_ids,
        sentence_ids=sent_ids,
        task_ids=task_ids,
        input_mask=input_mask,
        config=ernie_config,
        use_fp16=args.use_fp16)

    seq_out = ernie.get_sequence_output()
    cls_feats = ernie.get_pooled_output()
    # dummy layers to name the latent layers. the save_inf_model produce uncomprehensible names
    # like 'save_infer_model/scale_1'
    seq_out = fluid.layers.scale(seq_out, scale=1.0, name='ernie_sequence_latent')
    cls_feats = fluid.layers.scale(cls_feats, scale=1.0, name='ernie_classification')

    for i, inp in enumerate(inputs):
        print(f'input[{i}]:', inp.name, inp.shape, inp.dtype)
    print('sequence_output  :', seq_out.name, seq_out.shape, seq_out.dtype)
    print('classifier_output:', cls_feats.name, cls_feats.shape, cls_feats.dtype)
    return inputs, [seq_out, cls_feats]


def convert(args):
    ernie_export_path = f'{args.ernie_path}/ernie_persistables.pkl'
    pretraining_params_path = f'{args.ernie_path}/paddle/params'
    ernie_config_path = f'{args.ernie_path}/paddle/ernie_config.json'
    ernie_vocab_path = f'{args.ernie_path}/paddle/vocab.txt'
    unzip_message = f"Please unzip ERNIE paddle param archive into {args.ernie_path}/paddle"
    if not os.path.exists(pretraining_params_path):
        print(f"{pretraining_params_path} does not exist.", file=sys.stderr)
        print(unzip_message, file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(ernie_config_path):
        print(f"{ernie_config_path} does not exist.", file=sys.stderr)
        print(unzip_message, file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(ernie_vocab_path):
        print(f"{ernie_vocab_path} does not exist.", file=sys.stderr)
        print(unzip_message, file=sys.stderr)
        sys.exit(1)

    ernie_config = ErnieConfig(ernie_config_path)
    # Fix missing use_task_id
    ernie_config._config_dict['use_task_id'] = True
    ernie_config.print_config()

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)


    startup_prog = fluid.Program()
    train_program = fluid.Program()

    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        with fluid.program_guard(train_program, startup_prog):
            with fluid.unique_name.guard():
                _ = create_model(args, ernie_config=ernie_config)

                init_pretraining_params(
                    exe,
                    pretraining_params_path,
                    main_program=startup_prog,
                    use_fp16=args.use_fp16)
                persistables = dict()
                for var in filter(fluid.io.is_persistable, train_program.list_vars()):
                    numpy_value = fetch_var(var.name, inference_scope)
                    persistables[var.name] = numpy_value
                    if args.verbose:
                        print(var.name)
                print("totally", len(persistables), "persistables")
                with open(ernie_export_path, 'wb') as f:
                    pickle.dump(persistables, f)
    return train_program


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Export the ERNIE model in paddle export format')
    parser.add_argument('--ernie_path',
                        help='the directory containing artifacts of the ERNIE model. This is where you should unzip the downloaded ERNIE model archive.')
    parser.add_argument('--max_seq_len', type=int, default=512, help='Max sequence length')
    parser.add_argument('--use_fp16', type=bool, default=False, help='use 16-bit floating point numbers')
    parser.add_argument('--verbose', type=bool, default=False, help='Print verbose information')

    args = parser.parse_args(sys.argv[1:])
    convert(args)
