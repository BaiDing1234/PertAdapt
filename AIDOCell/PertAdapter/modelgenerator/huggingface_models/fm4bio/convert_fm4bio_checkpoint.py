####################################################################################################

# Copyright (c) 2021-, NVIDIA CORPORATION.  All rights reserved.
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

####################################################################################################

#
# Note: If when running this conversion script you're getting an exception:
#     ModuleNotFoundError: No module named 'megatron.model.enums'
# you need to tell python where to find the clone of Megatron-LM, e.g.:
#
# cd /tmp
# git clone https://github.com/NVIDIA/Megatron-LM
# PYTHONPATH=/tmp/Megatron-LM python src/transformers/models/fm4bio/convert_fm4bio_checkpoint.py ...
#
# if you already have it cloned elsewhere, simply adjust the path to the existing path
#
# If the training was done using a Megatron-LM fork, e.g.,
# https://github.com/microsoft/Megatron-DeepSpeed/ then chances are that you need to have that one
# in your path, i.e., /path/to/Megatron-DeepSpeed/
#

import argparse
import os
import re
import zipfile
import copy
import sys

import torch
from . import FM4BioModel, FM4BioConfig, FM4BioForMaskedLM
import warnings

####################################################################################################


def recursive_print(name, val, spaces=0):
    # Format the message.
    if name is None:
        msg = None
    else:
        fmt = "." * max(0, spaces - 2) + "# {:" + str(50 - spaces) + "s}"
        msg = fmt.format(name)

    # Print and recurse (if needed).
    if isinstance(val, dict):
        if msg is not None:
            print(msg)
        for k in val.keys():
            recursive_print(k, val[k], spaces + 2)
    elif isinstance(val, torch.Tensor):
        print(msg, ":", val.size())
    else:
        print(msg, ":", val)


def fix_query_key_value_ordering(
    param, checkpoint_version, num_splits, num_heads, hidden_size
):
    # Permutes layout of param tensor to [num_splits * num_heads * hidden_size, :]
    # for compatibility with later versions of NVIDIA Megatron-LM.
    # The inverse operation is performed inside Megatron-LM to read checkpoints:
    # https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/checkpointing.py#L209
    # If param is the weight tensor of the self-attention block, the returned tensor
    # will have to be transposed one more time to be read by HuggingFace BERT.
    input_shape = param.size()
    if checkpoint_version == 1.0:
        # version 1.0 stores [num_heads * hidden_size * num_splits, :]
        saved_shape = (num_heads, hidden_size, num_splits) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # other versions store [num_heads * num_splits * hidden_size, :]
        saved_shape = (num_heads, num_splits, hidden_size) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param


####################################################################################################


def convert_megatron_checkpoint_deepspeed(args, input_state_dict, config):

    # The converted output model.
    output_state_dict = {}
    megatron_args = input_state_dict["args"]

    # old versions did not store training args
    ds_args = input_state_dict.get("args", None)
    # if ds_args is not None:
    #     # do not make the user write a config file when the exact dimensions/sizes are already in the checkpoint
    #     # from pprint import pprint
    #     # pprint(vars(ds_args))

    #     config.tokenizer_type = ds_args.tokenizer_type
    #     config.vocab_size = ds_args.padded_vocab_size
    #     config.max_position_embeddings = ds_args.max_position_embeddings
    #     config.hidden_size = ds_args.hidden_size
    #     config.num_hidden_layers = ds_args.num_layers
    #     config.num_attention_heads = ds_args.num_attention_heads
    #     config.intermediate_size = ds_args.ffn_hidden_size if "ffn_hidden_size" in ds_args else 4 * ds_args.hidden_size
    #     # pprint(config)

    # The number of heads.
    heads = config.num_attention_heads
    # The hidden_size per head.
    hidden_size_per_head = config.hidden_size // heads
    # Megatron-LM checkpoint version
    if "checkpoint_version" in input_state_dict.keys():
        checkpoint_version = input_state_dict["checkpoint_version"]
    else:
        checkpoint_version = 0.0

    # The model.
    model = input_state_dict["module"]
    # The language model.
    lm = model["language_model"]
    # The embeddings.
    embeddings = lm["embedding"]

    # The word embeddings.
    word_embeddings = embeddings["word_embeddings"]["weight"]
    # Truncate the embedding table to vocab_size rows.
    word_embeddings = word_embeddings[: config.vocab_size, :]
    # Store the word embeddings.
    output_state_dict["bert.embeddings.word_embeddings.weight"] = word_embeddings

    if config.position_embedding_type != "rope":
        # The position embeddings.
        pos_embeddings = embeddings["position_embeddings"]["weight"]
        assert (
            pos_embeddings.size(0) == config.max_position_embeddings
            and pos_embeddings.size(1) == config.hidden_size
        )
        # Store the position embeddings.
        output_state_dict["bert.embeddings.position_embeddings.weight"] = pos_embeddings

        # The token-type embeddings.
        # tokentype_embeddings = embeddings["tokentype_embeddings"]["weight"]
        # Store the position embeddings.
        # output_state_dict["bert.embeddings.token_type_embeddings.weight"] = tokentype_embeddings

    # The transformer.
    transformer = lm["transformer"] if "transformer" in lm.keys() else lm["encoder"]

    # The regex to extract layer names.
    layer_re = re.compile(r"layers\.(\d+)\.([a-z0-9_.]+)\.([a-z_]+)")

    # The simple map of names for "automated" rules.
    megatron_to_transformers = {
        "attention.dense": ".attention.output.dense.",
        "self_attention.dense": ".attention.output.dense.",
        # "mlp.dense_h_to_4h": ".mlp.h_to_4h.", # should break down later
        # "mlp.dense_4h_to_h": ".mlp.down_proj.",
        "input_norm": ".attention.ln.",
        "post_attention_norm": ".ln.",
    }

    # Keep track of the attention/query/value tensor.
    attention_qkv_weight = None

    # Extract the layers.
    for key, val in transformer.items():
        # Match the name.
        m = layer_re.match(key)

        # Stop if that's not a layer
        if m is None:
            continue

        # The index of the layer.
        layer_idx = int(m.group(1))
        # The name of the operation.
        op_name = m.group(2)
        # Is it a weight or a bias?
        weight_or_bias = m.group(3)

        # The name of the layer.
        layer_name = f"bert.encoder.layer.{layer_idx}"

        # For layernorm(s), simply store the layer norm.
        if op_name.endswith("layernorm"):
            ln_name = "attention.ln" if op_name.startswith("input") else "ln"
            output_state_dict[layer_name + "." + ln_name + "." + weight_or_bias] = val

        # Transpose the QKV matrix.
        elif (
            op_name == "attention.query_key_value"
            or op_name == "self_attention.query_key_value"
        ) and weight_or_bias == "weight":
            # Make sure the QKV pointer is nil.
            assert attention_qkv_weight is None, ""

            out_val = fix_query_key_value_ordering(
                val, checkpoint_version, 3, heads, hidden_size_per_head
            )
            # Store the tensor as we need the bias as well to interleave QKV and biases.
            attention_qkv_weight = out_val

            # if not megatron_args.add_bias_linear:
            #     # handle it now

            #     # Split the QKV matrix into Q, K and V. Megatron stores Q,K,V interleaved.
            #     q = attention_qkv_weight[0 * config.hidden_size : 1 * config.hidden_size, :]
            #     k = attention_qkv_weight[1 * config.hidden_size : 2 * config.hidden_size, :]
            #     v = attention_qkv_weight[2 * config.hidden_size : 3 * config.hidden_size, :]

            #     # Store.
            #     output_state_dict[f"{layer_name}.attention.self.query.weight"] = q
            #     output_state_dict[f"{layer_name}.attention.self.key.weight"] = k
            #     output_state_dict[f"{layer_name}.attention.self.value.weight"] = v

            #     # Clear the stored tensor.
            #     attention_qkv_weight = None

        # Transpose the bias.
        elif (
            op_name == "attention.query_key_value"
            or op_name == "self_attention.query_key_value"
        ) and weight_or_bias == "bias":
            # Make sure we read the weight tensor.
            assert attention_qkv_weight is not None, ""

            # Split the QKV matrix into Q, K and V. Megatron stores Q,K,V interleaved.
            q = attention_qkv_weight[0 * config.hidden_size : 1 * config.hidden_size, :]
            k = attention_qkv_weight[1 * config.hidden_size : 2 * config.hidden_size, :]
            v = attention_qkv_weight[2 * config.hidden_size : 3 * config.hidden_size, :]

            out_val = fix_query_key_value_ordering(
                val, checkpoint_version, 3, heads, hidden_size_per_head
            )
            # Split the bias.
            q_bias = out_val[0 * config.hidden_size : 1 * config.hidden_size]
            k_bias = out_val[1 * config.hidden_size : 2 * config.hidden_size]
            v_bias = out_val[2 * config.hidden_size : 3 * config.hidden_size]

            # Store.
            output_state_dict[f"{layer_name}.attention.self.query.weight"] = q
            output_state_dict[f"{layer_name}.attention.self.query.bias"] = q_bias
            output_state_dict[f"{layer_name}.attention.self.key.weight"] = k
            output_state_dict[f"{layer_name}.attention.self.key.bias"] = k_bias
            output_state_dict[f"{layer_name}.attention.self.value.weight"] = v
            output_state_dict[f"{layer_name}.attention.self.value.bias"] = v_bias

            # Clear the stored tensor.
            attention_qkv_weight = None

        # Copy weights and biases as is.
        elif weight_or_bias in ["weight", "bias"]:
            if "dense_h_to_4h" in op_name:
                mlp_re = (
                    "mlp.deepspeed_moe.experts.deepspeed_experts.(\d+).dense_h_to_4h"
                )
                # target: ['bert.encoder.layer.X.mlp.dense_4h_to_h_Y.weight'
                expert_id = re.match(mlp_re, op_name).group(1)

                output_state_dict[
                    layer_name
                    + ".mlp.dense_h_to_4h_"
                    + expert_id
                    + "."
                    + weight_or_bias
                ] = val
            elif "dense_4h_to_h" in op_name:
                mlp_re = (
                    "mlp.deepspeed_moe.experts.deepspeed_experts.(\d+).dense_4h_to_h"
                )
                # target: ['bert.encoder.layer.X.mlp.dense_4h_to_h_Y.weight'
                expert_id = re.match(mlp_re, op_name).group(1)

                output_state_dict[
                    layer_name
                    + ".mlp.dense_4h_to_h_"
                    + expert_id
                    + "."
                    + weight_or_bias
                ] = val
            elif op_name == "mlp.gate.gate":
                output_state_dict[layer_name + ".mlp.router." + weight_or_bias] = val
            else:
                out_name = megatron_to_transformers[op_name]
                output_state_dict[layer_name + out_name + weight_or_bias] = val
        else:
            if weight_or_bias == "inv_freq":
                continue  # ignore
            raise ValueError(f"Unknown source: {key}")

    # The final layernorm.
    output_state_dict["bert.encoder.ln.weight"] = transformer["final_layernorm.weight"]
    # if megatron_args.add_bias_linear:
    #     output_state_dict["bert.encoder.ln.bias"] = transformer["final_norm.bias"]

    # # The pooler.
    # pooler = lm["pooler"]

    # # Store the matrix and the bias.
    # output_state_dict["bert.pooler.dense.weight"] = pooler["dense.weight"]
    # output_state_dict["bert.pooler.dense.bias"] = pooler["dense.bias"]

    # The LM head from Megatron (for RACE).
    if False:  # disable LM head
        lm_head = lm["lm_head"]

        # The transform matrix.
        output_state_dict["cls.predictions.transform.dense.weight"] = lm_head[
            "dense.weight"
        ]
        output_state_dict["cls.predictions.transform.dense.bias"] = lm_head[
            "dense.bias"
        ]

        # The transform LN.
        output_state_dict["cls.predictions.transform.LayerNorm.weight"] = lm_head[
            "norm.weight"
        ]
        if "norm.bias" in lm_head:
            output_state_dict["cls.predictions.transform.LayerNorm.bias"] = lm_head[
                "norm.bias"
            ]

        # For the decoder, we replicate the weights.
        output_state_dict["cls.predictions.decoder.weight"] = word_embeddings
        output_state_dict["cls.predictions.bias"] = lm_head["bias"]

        output_state_dict["cls.predictions.decoder.bias"] = output_state_dict[
            "cls.predictions.bias"
        ]
    else:
        output_state_dict["output_embed.weight"] = lm["output_layer"]["weight"]

    # The classifier from Megatron (for MLNI).
    # binary_head = model["binary_head"]

    # Store the classifier.
    # output_state_dict["cls.seq_relationship.weight"] = binary_head["weight"]
    # output_state_dict["cls.seq_relationship.bias"] = binary_head["bias"]

    # It should be done!
    return output_state_dict


def convert_megatron_checkpoint(args, input_state_dict, config, deepspeed=False):

    if deepspeed:
        return convert_megatron_checkpoint_deepspeed(args, input_state_dict, config)

    # The converted output model.
    output_state_dict = {}
    megatron_args = input_state_dict["args"]
    is_mcore_model = input_state_dict["args"].use_mcore_models

    # old versions did not store training args
    ds_args = input_state_dict.get("args", None)
    if ds_args is not None:
        # do not make the user write a config file when the exact dimensions/sizes are already in the checkpoint
        # from pprint import pprint
        # pprint(vars(ds_args))

        config.tokenizer_type = ds_args.tokenizer_type
        config.vocab_size = ds_args.padded_vocab_size
        config.max_position_embeddings = ds_args.max_position_embeddings
        config.hidden_size = ds_args.hidden_size
        config.num_hidden_layers = ds_args.num_layers
        config.num_attention_heads = ds_args.num_attention_heads
        config.intermediate_size = (
            ds_args.ffn_hidden_size
            if "ffn_hidden_size" in ds_args
            else 4 * ds_args.hidden_size
        )
        # pprint(config)

    # The number of heads.
    heads = config.num_attention_heads
    # The hidden_size per head.
    hidden_size_per_head = config.hidden_size // heads
    # Megatron-LM checkpoint version
    if "checkpoint_version" in input_state_dict.keys():
        checkpoint_version = input_state_dict["checkpoint_version"]
    else:
        checkpoint_version = 0.0

    if is_mcore_model:
        model = input_state_dict["model"]
        output_state_dict["bert.embeddings.word_embeddings.weight"] = model.pop(
            "embedding.word_embeddings.weight"
        )

        if ds_args.position_embedding_type != "rope":
            # The position embeddings.
            pos_embeddings = model.pop("embedding.position_embeddings.weight")
            assert (
                pos_embeddings.size(0) == config.max_position_embeddings
                and pos_embeddings.size(1) == config.hidden_size
            )
            # Store the position embeddings.
            output_state_dict["bert.embeddings.position_embeddings.weight"] = (
                pos_embeddings
            )

        attention_qkv_weight = None

        for layer_idx in range(ds_args.num_layers):
            source_target_mapping = {
                "encoder.layers.{layer_idx}.self_attention.linear_proj.weight": "bert.encoder.layer.{layer_idx}.attention.output.dense.weight",
                "encoder.layers.{layer_idx}.self_attention.linear_proj.bias": "bert.encoder.layer.{layer_idx}.attention.output.dense.bias",
                "encoder.layers.{layer_idx}.self_attention.linear_qkv.layer_norm_weight": "bert.encoder.layer.{layer_idx}.attention.ln.weight",
                "encoder.layers.{layer_idx}.self_attention.linear_qkv.layer_norm_bias": "bert.encoder.layer.{layer_idx}.attention.ln.bias",
                "encoder.layers.{layer_idx}.self_attention.linear_qkv.weight": "BREAKDOWN",  # should break down later
                "encoder.layers.{layer_idx}.self_attention.linear_qkv.bias": "BREAKDOWN",  # should break down later
                "encoder.layers.{layer_idx}.mlp.linear_fc1.layer_norm_weight": "bert.encoder.layer.{layer_idx}.ln.weight",
                "encoder.layers.{layer_idx}.mlp.linear_fc1.layer_norm_bias": "bert.encoder.layer.{layer_idx}.ln.bias",
                "encoder.layers.{layer_idx}.mlp.linear_fc1.weight": "BREAKDOWN",  # should break down later
                "encoder.layers.{layer_idx}.mlp.linear_fc1.bias": "BREAKDOWN",  # should break down later
                "encoder.layers.{layer_idx}.mlp.linear_fc2.weight": "bert.encoder.layer.{layer_idx}.mlp.down_proj.weight",
                "encoder.layers.{layer_idx}.mlp.linear_fc2.bias": "bert.encoder.layer.{layer_idx}.mlp.down_proj.bias",
            }

            for source, target in source_target_mapping.items():
                # there are 4 types of keys in the model that we need to handle:
                # qkv.weight, qkv.bias, fc1.weight, fc1.bias
                if target == "BREAKDOWN":
                    if "linear_qkv.weight" in source:
                        # Make sure the QKV pointer is nil.
                        assert attention_qkv_weight is None, ""
                        val = model.pop(source.format(layer_idx=layer_idx))
                        out_val = fix_query_key_value_ordering(
                            val, checkpoint_version, 3, heads, hidden_size_per_head
                        )
                        # Store the tensor as we need the bias as well to interleave QKV and biases.
                        attention_qkv_weight = out_val

                        if not megatron_args.add_bias_linear:
                            # handle it now

                            # Split the QKV matrix into Q, K and V. Megatron stores Q,K,V interleaved.
                            q = attention_qkv_weight[
                                0 * config.hidden_size : 1 * config.hidden_size, :
                            ]
                            k = attention_qkv_weight[
                                1 * config.hidden_size : 2 * config.hidden_size, :
                            ]
                            v = attention_qkv_weight[
                                2 * config.hidden_size : 3 * config.hidden_size, :
                            ]

                            # Store.
                            layer_name = f"bert.encoder.layer.{layer_idx}"
                            output_state_dict[
                                f"bert.encoder.layer.{layer_idx}.attention.self.query.weight"
                            ] = q
                            output_state_dict[
                                f"bert.encoder.layer.{layer_idx}.attention.self.key.weight"
                            ] = k
                            output_state_dict[
                                f"bert.encoder.layer.{layer_idx}.attention.self.value.weight"
                            ] = v

                            # Clear the stored tensor.
                            attention_qkv_weight = None
                    elif "linear_qkv.bias" in source:
                        # Make sure we read the weight tensor.
                        assert attention_qkv_weight is not None, ""

                        # Split the QKV matrix into Q, K and V. Megatron stores Q,K,V interleaved.
                        q = attention_qkv_weight[
                            0 * config.hidden_size : 1 * config.hidden_size, :
                        ]
                        k = attention_qkv_weight[
                            1 * config.hidden_size : 2 * config.hidden_size, :
                        ]
                        v = attention_qkv_weight[
                            2 * config.hidden_size : 3 * config.hidden_size, :
                        ]

                        val = model.pop(source.format(layer_idx=layer_idx))

                        out_val = fix_query_key_value_ordering(
                            val, checkpoint_version, 3, heads, hidden_size_per_head
                        )
                        # Split the bias.
                        q_bias = out_val[
                            0 * config.hidden_size : 1 * config.hidden_size
                        ]
                        k_bias = out_val[
                            1 * config.hidden_size : 2 * config.hidden_size
                        ]
                        v_bias = out_val[
                            2 * config.hidden_size : 3 * config.hidden_size
                        ]

                        # Store.
                        output_state_dict[
                            f"bert.encoder.layer.{layer_idx}.attention.self.query.weight"
                        ] = q
                        output_state_dict[
                            f"bert.encoder.layer.{layer_idx}.attention.self.query.bias"
                        ] = q_bias
                        output_state_dict[
                            f"bert.encoder.layer.{layer_idx}.attention.self.key.weight"
                        ] = k
                        output_state_dict[
                            f"bert.encoder.layer.{layer_idx}.attention.self.key.bias"
                        ] = k_bias
                        output_state_dict[
                            f"bert.encoder.layer.{layer_idx}.attention.self.value.weight"
                        ] = v
                        output_state_dict[
                            f"bert.encoder.layer.{layer_idx}.attention.self.value.bias"
                        ] = v_bias

                        # Clear the stored tensor.
                        attention_qkv_weight = None
                    elif "linear_fc1.weight" in source:
                        val = model.pop(source.format(layer_idx=layer_idx))
                        # Megatron stores the MLP as a single matrix, so we need to split it.
                        output_state_dict[
                            f"bert.encoder.layer.{layer_idx}.mlp.gate_proj.weight"
                        ] = val[: config.intermediate_size]
                        output_state_dict[
                            f"bert.encoder.layer.{layer_idx}.mlp.up_proj.weight"
                        ] = val[config.intermediate_size : 2 * config.intermediate_size]
                    elif "linear_fc1.bias" in source:
                        val = model.pop(source.format(layer_idx=layer_idx))
                        output_state_dict[
                            f"bert.encoder.layer.{layer_idx}.mlp.gate_proj.bias"
                        ] = val[: config.intermediate_size]
                        output_state_dict[
                            f"bert.encoder.layer.{layer_idx}.mlp.up_proj.bias"
                        ] = val[config.intermediate_size : 2 * config.intermediate_size]
                    else:
                        raise ValueError(f"Unknown source: {source}")

                else:
                    output_state_dict[target.format(layer_idx=layer_idx)] = model.pop(
                        source.format(layer_idx=layer_idx)
                    )

        # The final layernorm.
        output_state_dict["bert.encoder.ln.weight"] = model.pop(
            "encoder.final_layernorm.weight"
        )
        if megatron_args.add_bias_linear:
            output_state_dict["bert.encoder.ln.bias"] = model.pop(
                "encoder.final_layernorm.bias"
            )

        # lm head
        # The transform matrix.
        output_state_dict["cls.predictions.transform.dense.weight"] = model.pop(
            "lm_head.dense.weight"
        )
        output_state_dict["cls.predictions.transform.dense.bias"] = model.pop(
            "lm_head.dense.bias"
        )

        # The transform LN.
        output_state_dict["cls.predictions.transform.LayerNorm.weight"] = model.pop(
            "lm_head.layernorm.weight"
        )
        if "lm_head.layernorm.bias" in model:
            output_state_dict["cls.predictions.transform.LayerNorm.bias"] = model.pop(
                "lm_head.layernorm.bias"
            )

        if megatron_args.untie_embeddings_and_output_weights:
            raise NotImplementedError
        else:
            output_state_dict["cls.predictions.decoder.weight"] = output_state_dict[
                "bert.embeddings.word_embeddings.weight"
            ]  # lm_head.output_layer.weight is just a pointer to the word_embeddings
        output_state_dict["cls.predictions.bias"] = model.pop(
            "lm_head.output_layer.bias"
        )
        _ = model.pop(
            "output_layer.bias"
        )  # this is just a pointer to the lm_head.output_layer.bias
        output_state_dict["cls.predictions.decoder.bias"] = output_state_dict[
            "cls.predictions.bias"
        ]

        if len(model.keys()) > 0:
            warnings.warn(f"Unused keys in the model: {model.keys()}")

        return output_state_dict
        ########

        # # The regex to extract layer names.
        # layer_re = re.compile(r"encoder.layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

        # # The simple map of names for "automated" rules.
        # megatron_to_transformers = {
        #     # "attention.dense": ".attention.output.dense.",
        #     "self_attention.linear_proj": ".attention.output.dense.",
        #     "mlp.linear_fc1": ".mlp.h_to_4h.", # should break down later
        #     "mlp.linear_fc2": ".mlp.down_proj.",
        #     "input_norm": ".attention.ln.",
        #     "post_attention_norm": ".ln.",
        # }

        # # Keep track of the attention/query/value tensor.
        # attention_qkv_weight = None

        # # Extract the layers.
        # for key, val in model.items():
        #     # Match the name.
        #     m = layer_re.match(key)

        #     # skip if that's not a layer
        #     if m is None:
        #         continue

        #     # The index of the layer.
        #     layer_idx = int(m.group(1))
        #     # The name of the operation.
        #     op_name = m.group(2)
        #     # Is it a weight or a bias?
        #     weight_or_bias = m.group(3)

        #     # The name of the layer.
        #     layer_name = f"bert.encoder.layer.{layer_idx}"

        #     # For layernorm(s), simply store the layer norm.
        #     if op_name.endswith("layernorm"):
        #         ln_name = "attention.ln" if op_name.startswith("input") else "ln"
        #         output_state_dict[layer_name + "." + ln_name + "." + weight_or_bias] = val

        #     # Copy weights and biases as is.
        #     elif weight_or_bias in ["weight", "bias"]:
        #         if op_name == "mlp.dense_h_to_4h":
        #             # Megatron stores the MLP as a single matrix, so we need to split it.
        #             output_state_dict[layer_name + ".mlp.gate_proj." + weight_or_bias] = val[: config.intermediate_size]
        #             output_state_dict[layer_name + ".mlp.up_proj." + weight_or_bias] = val[config.intermediate_size : 2 * config.intermediate_size]
        #         else:
        #             out_name = megatron_to_transformers[op_name]
        #             output_state_dict[layer_name + out_name + weight_or_bias] = val

        # # The final layernorm.
        # output_state_dict["bert.encoder.ln.weight"] = model['encoder.final_layernorm.weight']
        # if megatron_args.add_bias_linear:
        #     output_state_dict["bert.encoder.ln.bias"] = model['encoder.final_layernorm.bias']

    else:
        # The model.
        model = input_state_dict["model"]
        # The language model.
        lm = model["language_model"]
        # The embeddings.
        embeddings = lm["embedding"]

        # The word embeddings.
        word_embeddings = embeddings["word_embeddings"]["weight"]
        # Truncate the embedding table to vocab_size rows.
        word_embeddings = word_embeddings[: config.vocab_size, :]
        # Store the word embeddings.
        output_state_dict["bert.embeddings.word_embeddings.weight"] = word_embeddings

        if ds_args.position_embedding_type != "rope":
            # The position embeddings.
            pos_embeddings = embeddings["position_embeddings"]["weight"]
            assert (
                pos_embeddings.size(0) == config.max_position_embeddings
                and pos_embeddings.size(1) == config.hidden_size
            )
            # Store the position embeddings.
            output_state_dict["bert.embeddings.position_embeddings.weight"] = (
                pos_embeddings
            )

            # The token-type embeddings.
            # tokentype_embeddings = embeddings["tokentype_embeddings"]["weight"]
            # Store the position embeddings.
            # output_state_dict["bert.embeddings.token_type_embeddings.weight"] = tokentype_embeddings

        # The transformer.
        transformer = lm["transformer"] if "transformer" in lm.keys() else lm["encoder"]

        # The regex to extract layer names.
        layer_re = re.compile(r"layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

        # The simple map of names for "automated" rules.
        megatron_to_transformers = {
            "attention.dense": ".attention.output.dense.",
            "self_attention.dense": ".attention.output.dense.",
            "mlp.dense_h_to_4h": ".mlp.h_to_4h.",  # should break down later
            "mlp.dense_4h_to_h": ".mlp.down_proj.",
            "input_norm": ".attention.ln.",
            "post_attention_norm": ".ln.",
        }

        # Keep track of the attention/query/value tensor.
        attention_qkv_weight = None

        # Extract the layers.
        for key, val in transformer.items():
            # Match the name.
            m = layer_re.match(key)

            # Stop if that's not a layer
            if m is None:
                break

            # The index of the layer.
            layer_idx = int(m.group(1))
            # The name of the operation.
            op_name = m.group(2)
            # Is it a weight or a bias?
            weight_or_bias = m.group(3)

            # The name of the layer.
            layer_name = f"bert.encoder.layer.{layer_idx}"

            # For layernorm(s), simply store the layer norm.
            if op_name.endswith("layernorm"):
                ln_name = "attention.ln" if op_name.startswith("input") else "ln"
                output_state_dict[layer_name + "." + ln_name + "." + weight_or_bias] = (
                    val
                )

            # Transpose the QKV matrix.
            elif (
                op_name == "attention.query_key_value"
                or op_name == "self_attention.query_key_value"
            ) and weight_or_bias == "weight":
                # Make sure the QKV pointer is nil.
                assert attention_qkv_weight is None, ""

                out_val = fix_query_key_value_ordering(
                    val, checkpoint_version, 3, heads, hidden_size_per_head
                )
                # Store the tensor as we need the bias as well to interleave QKV and biases.
                attention_qkv_weight = out_val

                if not megatron_args.add_bias_linear:
                    # handle it now

                    # Split the QKV matrix into Q, K and V. Megatron stores Q,K,V interleaved.
                    q = attention_qkv_weight[
                        0 * config.hidden_size : 1 * config.hidden_size, :
                    ]
                    k = attention_qkv_weight[
                        1 * config.hidden_size : 2 * config.hidden_size, :
                    ]
                    v = attention_qkv_weight[
                        2 * config.hidden_size : 3 * config.hidden_size, :
                    ]

                    # Store.
                    output_state_dict[f"{layer_name}.attention.self.query.weight"] = q
                    output_state_dict[f"{layer_name}.attention.self.key.weight"] = k
                    output_state_dict[f"{layer_name}.attention.self.value.weight"] = v

                    # Clear the stored tensor.
                    attention_qkv_weight = None

            # Transpose the bias.
            elif (
                op_name == "attention.query_key_value"
                or op_name == "self_attention.query_key_value"
            ) and weight_or_bias == "bias":
                # Make sure we read the weight tensor.
                assert attention_qkv_weight is not None, ""

                # Split the QKV matrix into Q, K and V. Megatron stores Q,K,V interleaved.
                q = attention_qkv_weight[
                    0 * config.hidden_size : 1 * config.hidden_size, :
                ]
                k = attention_qkv_weight[
                    1 * config.hidden_size : 2 * config.hidden_size, :
                ]
                v = attention_qkv_weight[
                    2 * config.hidden_size : 3 * config.hidden_size, :
                ]

                out_val = fix_query_key_value_ordering(
                    val, checkpoint_version, 3, heads, hidden_size_per_head
                )
                # Split the bias.
                q_bias = out_val[0 * config.hidden_size : 1 * config.hidden_size]
                k_bias = out_val[1 * config.hidden_size : 2 * config.hidden_size]
                v_bias = out_val[2 * config.hidden_size : 3 * config.hidden_size]

                # Store.
                output_state_dict[f"{layer_name}.attention.self.query.weight"] = q
                output_state_dict[f"{layer_name}.attention.self.query.bias"] = q_bias
                output_state_dict[f"{layer_name}.attention.self.key.weight"] = k
                output_state_dict[f"{layer_name}.attention.self.key.bias"] = k_bias
                output_state_dict[f"{layer_name}.attention.self.value.weight"] = v
                output_state_dict[f"{layer_name}.attention.self.value.bias"] = v_bias

                # Clear the stored tensor.
                attention_qkv_weight = None

            # Copy weights and biases as is.
            elif weight_or_bias in ["weight", "bias"]:
                if op_name == "mlp.dense_h_to_4h":
                    # Megatron stores the MLP as a single matrix, so we need to split it.
                    output_state_dict[
                        layer_name + ".mlp.gate_proj." + weight_or_bias
                    ] = val[: config.intermediate_size]
                    output_state_dict[layer_name + ".mlp.up_proj." + weight_or_bias] = (
                        val[config.intermediate_size : 2 * config.intermediate_size]
                    )
                else:
                    out_name = megatron_to_transformers[op_name]
                    output_state_dict[layer_name + out_name + weight_or_bias] = val

        # The final layernorm.
        output_state_dict["bert.encoder.ln.weight"] = transformer["final_norm.weight"]
        if megatron_args.add_bias_linear:
            output_state_dict["bert.encoder.ln.bias"] = transformer["final_norm.bias"]

        # # The pooler.
        # pooler = lm["pooler"]

        # # Store the matrix and the bias.
        # output_state_dict["bert.pooler.dense.weight"] = pooler["dense.weight"]
        # output_state_dict["bert.pooler.dense.bias"] = pooler["dense.bias"]

        # The LM head from Megatron (for RACE).
        lm_head = model["lm_head"]

        # The transform matrix.
        output_state_dict["cls.predictions.transform.dense.weight"] = lm_head[
            "dense.weight"
        ]
        output_state_dict["cls.predictions.transform.dense.bias"] = lm_head[
            "dense.bias"
        ]

        # The transform LN.
        output_state_dict["cls.predictions.transform.LayerNorm.weight"] = lm_head[
            "norm.weight"
        ]
        if "norm.bias" in lm_head:
            output_state_dict["cls.predictions.transform.LayerNorm.bias"] = lm_head[
                "norm.bias"
            ]

        # For the decoder, we replicate the weights.
        output_state_dict["cls.predictions.decoder.weight"] = word_embeddings
        output_state_dict["cls.predictions.bias"] = lm_head["bias"]

        output_state_dict["cls.predictions.decoder.bias"] = output_state_dict[
            "cls.predictions.bias"
        ]

        # The classifier from Megatron (for MLNI).
        # binary_head = model["binary_head"]

        # Store the classifier.
        # output_state_dict["cls.seq_relationship.weight"] = binary_head["weight"]
        # output_state_dict["cls.seq_relationship.bias"] = binary_head["bias"]

        # It should be done!
        return output_state_dict


####################################################################################################
import re
from collections import defaultdict


def load_mp_state_dict(base_dir, deepspeed_format=True):
    mp_list = os.listdir(base_dir)
    mp_list = [mp for mp in mp_list if mp.startswith("mp_rank_")]
    mp_list.sort()
    master_state_dict = {}
    print(f"mp_list: {mp_list}")  # mp_rank_00_000, mp_rank_00_001, ...

    if deepspeed_format:
        # find anything that matches 'layer_*_expert_*_mp_rank_*_model_states.pt'
        expert_list = os.listdir(base_dir)
        regex = re.compile(r"layer_(\d+)_expert_(\d+)_mp_rank_(\d+)_model_states.pt")
        expert_list = [mp for mp in expert_list if regex.match(mp)]
        print(f"expert_list: {expert_list}")

        # pipeline_parallel_cnt = len(
        #     mp_list
        # )
        for mp in mp_list:
            # pipeline_idx = int(mp.split('_')[-1])
            # print(f'mp_file: {mp}, pipeline_idx: {pipeline_idx}')
            x0 = torch.load(
                os.path.join(base_dir, mp), map_location=torch.device("cuda:0")
            )
            # del x0['optimizer']
            # x0 = x0['module']

            if len(master_state_dict) == 0:
                # the first checkpoint, just copy
                master_state_dict.update(x0)
                # don't need "optimizer"
            else:

                assert False
                # absorb x0['model']['language_model']['encoder']
                current_layers_keys = master_state_dict["model"]["language_model"][
                    "encoder"
                ].keys()
                # eg, 'layers.0.self_attention.query_key_value.weight'
                # get all prefix "layers.X"
                layer_indices = set(
                    [int(k.split(".")[1]) for k in current_layers_keys]
                )  # {0, 1, 2, 3, 4, 5, 6, 7, 8}
                next_layer_idx = max(layer_indices) + 1

                # change names: layers.X.self_attention.dense.weight -> layers.{X+next_layer_idx}.self_attention.dense.weight
                new_layers = {}
                for k, v in x0["model"]["language_model"]["encoder"].items():
                    if k.startswith("layers."):
                        local_layer_key = k.split(".")
                        local_layer_idx = int(local_layer_key[1])
                        global_layer_idx = local_layer_idx + next_layer_idx
                        global_key = copy.deepcopy(local_layer_key)
                        global_key[1] = str(global_layer_idx)
                        global_key = ".".join(global_key)

                        print(f"{mp}: {k} -> {global_key}")
                        new_layers[global_key] = v
                    else:
                        print(f"{mp}: {k}")
                        new_layers[k] = v

                # update master state dict
                master_state_dict["model"]["language_model"]["encoder"].update(
                    new_layers
                )

                # update other keys under "model"
                for k, v in x0["model"].items():
                    if k not in master_state_dict["model"]:
                        print(f"copy {k}")
                        master_state_dict["model"][k] = v
                    elif k == "language_model":
                        pass
                    else:
                        assert False, f"Duplicated key: {k}"

        if expert_list:  # not empty
            # create a dict to sort the expert_list
            # so that expert_dict [LAYER][EXPERT][RANK] = FILE
            expert_dict = defaultdict(lambda: defaultdict(dict))
            for mp in expert_list:
                layer, expert, rank = regex.match(mp).groups()
                # expert_dict[int(layer)][int(expert)][int(rank)] = mp
                # rename: language_model.encoder.layers.X --> layers.X
                mp_expert_dict = torch.load(
                    os.path.join(base_dir, mp), map_location=torch.device("cuda:0")
                )
                mp_expert_dict_renamed = {}
                for k, v in mp_expert_dict.items():
                    new_key = k.replace("language_model.encoder.layers", "layers")
                    mp_expert_dict_renamed[new_key] = v
                master_state_dict["module"]["language_model"]["encoder"].update(
                    mp_expert_dict_renamed
                )

        print("--" * 20)

        print("master_state_dict keys:")

        def recursive_print_keys(d, spaces=0):
            for k, v in d.items():
                if isinstance(v, dict):
                    print("." * spaces + k)
                    recursive_print_keys(v, spaces + 2)
                else:
                    print("." * spaces + k)

        recursive_print_keys(master_state_dict["module"])

        # for k in master_state_dict['module']['language_model']['encoder'].keys():
        #     print(k)
        # for k in master_state_dict['module']['language_model']['encoder'].keys():
        #     print(k)

        return master_state_dict
    else:
        # Megatron-LM format

        # find anything that matches 'layer_*_expert_*_mp_rank_*_model_states.pt'
        expert_list = os.listdir(base_dir)
        regex = re.compile(r"layer_(\d+)_expert_(\d+)_mp_rank_(\d+)_model_states.pt")
        expert_list = [mp for mp in expert_list if regex.match(mp)]
        print(f"expert_list: {expert_list}")

        if expert_list:  # not empty
            # create a dict to sort the expert_list
            # so that expert_dict [LAYER][EXPERT][RANK] = FILE
            expert_dict = defaultdict(lambda: defaultdict(dict))
            for mp in expert_list:
                layer, expert, rank = regex.match(mp).groups()
                expert_dict[int(layer)][int(expert)][int(rank)] = mp

        pipeline_parallel_cnt = len(mp_list)
        for mp in mp_list:
            pipeline_idx = int(mp.split("_")[-1])
            print(f"mp_file: {mp}, pipeline_idx: {pipeline_idx}")
            x0 = torch.load(
                os.path.join(base_dir, mp, "model_optim_rng.pt"),
                map_location=torch.device("cuda:0"),
            )
            del x0["optimizer"]

            if len(master_state_dict) == 0:
                # the first checkpoint, just copy
                master_state_dict.update(x0)
                # don't need "optimizer"
            else:
                # absorb x0['model']['language_model']['encoder']
                current_layers_keys = master_state_dict["model"]["language_model"][
                    "encoder"
                ].keys()
                # eg, 'layers.0.self_attention.query_key_value.weight'
                # get all prefix "layers.X"
                layer_indices = set(
                    [int(k.split(".")[1]) for k in current_layers_keys]
                )  # {0, 1, 2, 3, 4, 5, 6, 7, 8}
                next_layer_idx = max(layer_indices) + 1

                # change names: layers.X.self_attention.dense.weight -> layers.{X+next_layer_idx}.self_attention.dense.weight
                new_layers = {}
                for k, v in x0["model"]["language_model"]["encoder"].items():
                    if k.startswith("layers."):
                        local_layer_key = k.split(".")
                        local_layer_idx = int(local_layer_key[1])
                        global_layer_idx = local_layer_idx + next_layer_idx
                        global_key = copy.deepcopy(local_layer_key)
                        global_key[1] = str(global_layer_idx)
                        global_key = ".".join(global_key)

                        print(f"{mp}: {k} -> {global_key}")
                        new_layers[global_key] = v
                    else:
                        print(f"{mp}: {k}")
                        new_layers[k] = v

                # update master state dict
                master_state_dict["model"]["language_model"]["encoder"].update(
                    new_layers
                )

                # update other keys under "model"
                for k, v in x0["model"].items():
                    if k not in master_state_dict["model"]:
                        print(f"copy {k}")
                        master_state_dict["model"][k] = v
                    elif k == "language_model":
                        pass
                    else:
                        assert False, f"Duplicated key: {k}"

        print("--" * 20)

        print("master_state_dict keys:")
        for k in master_state_dict["model"]["language_model"]["encoder"].keys():
            print(k)

        return master_state_dict


def main():
    # Create the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-checkpoint-structure", action="store_true")
    parser.add_argument("--from-deepspeed", action="store_true")
    parser.add_argument(
        "path_to_checkpoint",
        type=str,
        help="Path to the ZIP file containing the checkpoint",
    )
    parser.add_argument(
        "output_path", type=str, help="Path to the hf model output directory"
    )
    parser.add_argument(
        "--config_file",
        default="",
        type=str,
        help="An optional config json file describing the pre-trained model.",
    )
    args = parser.parse_args()

    MEGATRON_PATH = "/home/tianhua.tao/bio/protein-moe/proteinglm_100b_abla_ddp_ds"

    # add to PATH because the deepspeed checkpoint has a different structure
    # it will need to import the module megatron
    sys.path.append(MEGATRON_PATH)

    # Extract the basename.
    basename = os.path.dirname(args.path_to_checkpoint)

    # Load the model.
    # the .zip is very optional, let's keep it for backward compatibility
    print(f'Extracting PyTorch state dictionary from "{args.path_to_checkpoint}"')
    if args.path_to_checkpoint.endswith(".zip"):
        with zipfile.ZipFile(args.path_to_checkpoint, "r") as checkpoint:
            with checkpoint.open(
                "release/mp_rank_00/model_optim_rng.pt"
            ) as pytorch_dict:
                input_state_dict = torch.load(pytorch_dict, map_location="cpu")
    else:
        # works with pipeline parallelism, no tensor parallelism
        input_state_dict = load_mp_state_dict(
            args.path_to_checkpoint, args.from_deepspeed
        )
        # input_state_dict = torch.load(args.path_to_checkpoint, map_location="cuda:0")

    megatron_args = input_state_dict["args"]

    if args.from_deepspeed:
        if args.config_file == "":
            # Default config of fm4bio 345m
            config = FM4BioConfig(
                vocab_size=input_state_dict["module"]["language_model"]["output_layer"][
                    "weight"
                ].size(0),
                num_hidden_layers=megatron_args.num_layers,
                hidden_size=megatron_args.hidden_size,
                num_attention_heads=megatron_args.num_attention_heads,
                max_position_embeddings=megatron_args.seq_length,
                intermediate_size=megatron_args.ffn_hidden_size,
                normalization_type=(
                    "RMSNorm"
                    if megatron_args.normalization == "rmsnorm"
                    else "LayerNorm"
                ),
                add_linear_bias=True,
                position_embedding_type="rope",
                moe=megatron_args.num_experts[0] > 1,
                num_experts=megatron_args.num_experts[0],
                experts_per_token=megatron_args.topk,
                use_lm_head=False,  # the BERT in megatron-deepspeed is implemented with GPT actually, so it does not have lm head
                tie_word_embeddings=False,
            )

        else:
            config = FM4BioConfig.from_json_file(args.config_file)

    else:
        is_mcore_model = input_state_dict["args"].use_mcore_models

        if args.config_file == "":
            # Default config of fm4bio 345m
            config = FM4BioConfig(
                vocab_size=128,
                num_hidden_layers=megatron_args.encoder_num_layers,
                hidden_size=megatron_args.hidden_size,
                num_attention_heads=megatron_args.num_attention_heads,
                max_position_embeddings=megatron_args.max_position_embeddings,
                intermediate_size=megatron_args.ffn_hidden_size,
                normalization_type=megatron_args.normalization,
                add_linear_bias=megatron_args.add_linear_bias,
                position_embedding_type=(
                    "rope"
                    if megatron_args.use_rotary_position_embeddings
                    else "absolute"
                ),
            )

            # different fm4bio-*-345m models have different vocab sizes, so override the default
            # config (which is for fm4bio-cased-345m) with the actual vocab dimension
            if is_mcore_model:
                config.vocab_size = input_state_dict["model"]["output_layer.bias"].size(
                    0
                )
            else:
                config.vocab_size = input_state_dict["model"]["lm_head"]["bias"].numel()
        else:
            config = FM4BioConfig.from_json_file(args.config_file)

    model = FM4BioForMaskedLM(config)
    hf_state_dict = model.state_dict()
    print(hf_state_dict.keys())

    # Convert.
    print("Converting")
    output_state_dict = convert_megatron_checkpoint(
        args, input_state_dict, config, args.from_deepspeed
    )

    # Print the structure of converted state dict.
    if args.print_checkpoint_structure:
        recursive_print(None, output_state_dict)

    # compare hf_state_dict and output_state_dict
    for k in hf_state_dict.keys():
        if k not in output_state_dict.keys():
            print(f'key "{k}" not found in output_state_dict')
            continue

        if hf_state_dict[k].size() != output_state_dict[k].size():
            print(
                f'key "{k}" has different size: {hf_state_dict[k].size()} vs {output_state_dict[k].size()}'
            )

    for k in output_state_dict.keys():
        if k not in hf_state_dict.keys():
            print(f'key "{k}" not found in hf_state_dict')

    debug_meg = output_state_dict["bert.embeddings.word_embeddings.weight"].clone()
    debug_in_before = model.state_dict()[
        "bert.embeddings.word_embeddings.weight"
    ].clone()
    debug_out_before = model.state_dict()["output_embed.weight"].clone()
    model.load_state_dict(output_state_dict)
    debug_in_after = model.state_dict()[
        "bert.embeddings.word_embeddings.weight"
    ].clone()
    debug_out_after = model.state_dict()["output_embed.weight"].clone()

    print("Saving model to", args.output_path)
    model.save_pretrained(args.output_path, safe_serialization=False)
    # Store the config to file.

    # config.save_pretrained(basename)

    # # Store the state_dict to file.
    # output_checkpoint_file = os.path.join(basename, "pytorch_model.bin")
    # print(f'Saving checkpoint to "{output_checkpoint_file}"')
    # torch.save(output_state_dict, output_checkpoint_file)


####################################################################################################

if __name__ == "__main__":
    main()

####################################################################################################
