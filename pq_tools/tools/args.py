# -*- coding=utf-8 -*-
"""
#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import six
import argparse


def str2bool(v):
    """
    str2bool
    """
    # because argparse does not support to parse "true, False" as python
    # boolean directly
    return v.lower() in ("true", "t", "1")


class ArgumentGroup(object):
    """
    ArgumentGroup
    """

    def __init__(self, parser, title, des):
        """
        init
        """
        self._group = parser.add_argument_group(title=title, description=des)

    def add_arg(self, name, type, default, help, **kwargs):
        """
        add_arg
        """
        type = str2bool if type == bool else type
        # if type == list: # by dwk
        #     self._group.add_argument("--" + name, nargs='+', type=int)
        # else:
        self._group.add_argument(
            "--" + name,
            default=default,
            type=type,
            help=help + ' Default: %(default)s.',
            **kwargs)


def parse_args():
    """
    parse_args
    """
    # global
    parser = argparse.ArgumentParser("main")

    # model
    model_g = ArgumentGroup(
        parser, "model", "options to init, resume and save model.")
    model_g.add_arg("epoch_num", int, 3, "number of epochs for train")
    model_g.add_arg("batch_size", int, 16, "batch size for train")
    model_g.add_arg("learning_rate", float, 1e-4,
                    "learning rate for global training")
    model_g.add_arg("node_emb_size", int, 256, "node embedding size")
    model_g.add_arg("path_emb_size", int, 4, "path embedding size")
    model_g.add_arg("path_indepent_layer_size", int, 6, "path embedding size")
    model_g.add_arg("query_emb_size", int, 256, "input query embedding size")
    model_g.add_arg("topK", int, 1, "best recall result nums")
    model_g.add_arg("nce_num", int, 5, "nce negtive sample nums")
    model_g.add_arg("K", int, 5, "graph node nums")
    model_g.add_arg("D", int, 8, "layer size")
    model_g.add_arg("neg_sampling_strategy", str, '', "negtive sampling strategy")
    model_g.add_arg("neg_sampling_exp_base", float, 1.0, "negtive sampling base numble")
    model_g.add_arg("sub_path", bool, False, "use logits of sub path to compute loss")
    model_g.add_arg("beam_search_strategy", str, 'viterbi', "beam search strategy for inference")
    model_g.add_arg("pair_wise", bool, False, "pairwise")

    model_g = ArgumentGroup(
        parser, "path", "files path of data & model.")
    model_g.add_arg("train_files_path", str, "./data/train", "train data path")
    model_g.add_arg("test_files_path", str, "./data/test", "test data path")
    model_g.add_arg("infer_files_path", str, "./data/infer", "test data path")
    model_g.add_arg("checkpoint_path", str, "./ckpt", "test data path")
    model_g.add_arg("load_checkpoint_path", str, "", "load checkpoint")
    model_g.add_arg("infer_model", str, "./ckpt/ckpt10000", 'infer model')
    model_g.add_arg("is_infer", bool, False, "is infer")
    model_g.add_arg("is_infer_withdot", bool, False, "is infer")
    model_g.add_arg("is_deploy", bool, False, "is deploy")
    model_g.add_arg("deploy_model_save_path", str, "", "deploy model save path")
    model_g.add_arg("infer_device_id", int, 0, "device id")
    model_g.add_arg("infer_device_count", int, 1, "device count")
    model_g.add_arg("is_predict", bool, False, "is infer")
    model_g.add_arg("is_distributed", bool, False, "is distributed")
    model_g.add_arg("use_cuda", bool, True, "is distributed")
    model_g.add_arg("use_fast_executor", bool, False, "use experimental executor")
    model_g.add_arg("max_queue", int, 64, "max_queue of multiprocessing in data reader")
    model_g.add_arg("num_workers", int, 1, "num_workers of multiprocessing in data reader")
    model_g.add_arg("pid2path_file", str, "", "test data path")
    model_g.add_arg("pid_emb_path", str, "", "test data path")
    model_g.add_arg("is_validation_task", bool, False, "is validation task")
    
    # build graph and warm up
    model_g.add_arg("pid_emb_fp", str,
                    "../data/pid_title_emb.txt", "pid_emb_fp file path")
    model_g.add_arg("ground_truth", str,
                    "None", "ground_truth file path")
    model_g.add_arg("query2pid_fp", str,
                    "../data/query2pid.txt", "query2pid_fp file path")
    model_g.add_arg("pid2pqindex_fp", str,
                    "../data/pid2pqindex.txt", "pid2pqindex_fp file path")
    model_g.add_arg("query_emb_fp", str,
                    "../data/query_emb.txt", "query_emb_fp file path")
    model_g.add_arg("pid2title_fp", str,
                    "../data/pid_title.txt", "pid_title_fp file path")
    model_g.add_arg("train_fp", str,
                    "../data/train.tsv", "train_fp file path")
    model_g.add_arg("test_fp", str,
                    "../data/test.tsv", "test_fp file path")
    model_g.add_arg("skip_steps", int, 50, "step num for skipping fectch")
    model_g.add_arg("show_steps", int, 20, "step num for show info")
    model_g.add_arg("validation_steps", int, 1000, "step num for validaiton")
    model_g.add_arg("steps_per_epoch", int, 1000, "step num for validaiton")
    model_g.add_arg("save_steps", int, 5000, "step num for saving checkpoint")

    # build transformer
    model_g.add_arg("ernie_config_path", str, "./tools/ernie_distill_student.json", "")
    
    args = parser.parse_args()
    return args


def print_arguments(args):
    """
    print arguments
    """
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')
