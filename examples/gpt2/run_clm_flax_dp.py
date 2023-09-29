#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
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
Pre-training/Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import json
import logging
import math
import os
#import objsize
os.environ['no_proxy'] = '*'
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
import functools
from itertools import chain
from pathlib import Path
from typing import Callable, Optional

import datasets
import numpy as np
from datasets import Dataset, load_dataset
from tqdm import tqdm

import alpa
from alpa.model.model_util import DynamicScale, TrainState
import jax
import jax.numpy as jnp
import optax
import transformers
from flax import jax_utils, traverse_util
from flax.training import train_state
from flax.training.common_utils import onehot, shard, shard_prng_key
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    FLAX_MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    FlaxAutoModelForCausalLM,
    HfArgumentParser,
    is_tensorboard_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.utils import get_full_repo_name, send_example_telemetry
import ray
from collections import namedtuple
# from clu import parameter_overview

#os.environ["NCCL_DEBUG"] = "INFO"
#os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
#os.environ["NCCL_P2P_DISABLE"] = "1"
#os.environ["NCCL_LAUNCH_MODE"] = "PARALLEL"
#ray.shutdown()
# cmd에서 Ray start --head를 하면 안됨.

ray.init(log_to_driver=False, _plasma_directory="/tmp", _system_config={
        "automatic_object_spilling_enabled": True,
        "object_spilling_config": json.dumps(
            {"type": "filesystem", "params": {"directory_path": "/tmp/spill"}},
        )
    },
)
alpa.init(cluster="ray")
# [TODO] ray.shutdown()


#alpa.init(cluster="ray")
#tf.config.experimental.set_visible_devices([], 'GPU')

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(FLAX_MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class TrainingArguments:
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    num_micro_batches: int = field(default=1, metadata={"help": "The number of micro batches for gradient accumulation."})
    seqlen: int = field(default=1024,
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    hiddensize: int = field(default=768,
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    numlayers: int = field(default=12,
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    vocabsize: int = field(default=51200,
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    numheads: int = field(default=12,
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    layer_num: int = field(default=1, metadata={"help": "The number of micro batches for gradient accumulation."})
    mod: str = field(default="1.5B", metadata={"help": "The number of micro batches for gradient accumulation."})
    seq_len: int = field(default=1, metadata={"help": "The degree of operator model parallelism."})
    operator_parallel: int = field(default=1, metadata={"help": "The degree of operator model parallelism."})
    pipeline_parallel: int = field(default=1, metadata={"help": "The degree of pipeline model parallelism."})
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    adafactor: bool = field(default=False, metadata={"help": "Whether or not to replace AdamW by Adafactor."})
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."}) # for only learning rate decay
    logging_steps: int = field(default=500, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})
    eval_steps: int = field(default=None, metadata={"help": "Run an evaluation every X steps."})
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    push_to_hub: bool = field(
        default=False, metadata={"help": "Whether or not to upload the trained model to the model hub after training."}
    )
    hub_model_id: str = field(
        default=None, metadata={"help": "The name of the repository to keep in sync with the local `output_dir`."}
    )
    hub_token: str = field(default=None, metadata={"help": "The token to use to push to the Model Hub."})

    def __post_init__(self):
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": (
                "Floating-point format in which the model weights should be initialized and trained. Choose one of"
                " `[float32, float16, bfloat16]`."
            )
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def data_loader(rng: jax.random.PRNGKey, dataset: Dataset, batch_size: int,
                min_batch_size: int, shuffle: bool = False):
    """
    Returns batches of size `batch_size` from truncated `dataset`, sharded over all local devices.
    Shuffle batches if `shuffle` is `True`.
    """
    if len(dataset) < batch_size:
        assert len(dataset) >= min_batch_size
        batch_size = len(dataset) // min_batch_size * min_batch_size

    data_collator = transformers.DefaultDataCollator("np")
    tf_dataset = dataset.to_tf_dataset(batch_size=batch_size,
                                       columns=dataset.column_names,
                                       collate_fn=data_collator,
                                       shuffle=shuffle,
                                       drop_remainder=True)

    for batch in tf_dataset:
        batch = {k: v._numpy() for k, v in batch.items()}
        yield batch


def write_train_metric(summary_writer, train_metrics, train_time, step):
    summary_writer.scalar("train_time", train_time, step)

    train_metrics = alpa.util.get_metrics(train_metrics)
    for key, vals in train_metrics.items():
        tag = f"train_{key}"
        for i, val in enumerate(vals):
            summary_writer.scalar(tag, val, step - len(vals) + i + 1)


def create_learning_rate_fn(
    train_ds_size: int, train_batch_size: int, num_train_epochs: int, num_warmup_steps: int, learning_rate: float
) -> Callable[[int], jnp.array]:
    """Returns a linear warmup, linear_decay learning rate function."""
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps)
    decay_fn = optax.linear_schedule(
        init_value=learning_rate, end_value=0, transition_steps=num_train_steps - num_warmup_steps
    )
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
    return schedule_fn


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args, framework="flax")

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()

    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Handle the repository creation
    '''if training_args.push_to_hub: #default False
        if training_args.hub_model_id is None:
            repo_name = get_full_repo_name(
                Path(training_args.output_dir).absolute().name, token=training_args.hub_token
            )
        else:
            repo_name = training_args.hub_model_id
        repo = Repository(training_args.output_dir, clone_from=repo_name)'''

    #  Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    '''if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            keep_in_memory=False,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        if "validation" not in dataset.keys():
            dataset["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            dataset["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = data_args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        dataset = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            **dataset_args,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        if "validation" not in dataset.keys():
            dataset["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                **dataset_args,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            dataset["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                **dataset_args,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer

    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    print("dataset build OK================")

    if model_args.config_name: #"./norwegian-gpt2"
        config = AutoConfig.from_pretrained(
            model_args.config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            #hidden_size=5120,
            #num_attention_heads=40,
            #num_hidden_layers=40,
        )
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            #hidden_size=5120,
            #num_attention_heads=40,
            #num_hidden_layers=40,
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    print("get_toekenizer")
    print("model_args.cache_dir: ", model_args.cache_dir) # None
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            use_auth_token=True if model_args.use_auth_token else None,
            #hidden_size=5120,
            #um_attention_heads=40,
            #num_hidden_layers=40,
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            use_auth_token=True if model_args.use_auth_token else None,
            #hidden_size=5120,
            #num_attention_heads=40,
            #num_hidden_layers=40,
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    

    print("get pretrain_model")
    if model_args.model_name_or_path: # pre-trained model 들고와
        model = FlaxAutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            seed=training_args.seed,
            dtype=getattr(jnp, model_args.dtype),
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else: # 여기 들어옴.
        model = FlaxAutoModelForCausalLM.from_config(
            config,
            seed=training_args.seed,
            dtype=getattr(jnp, model_args.dtype),
            #hidden_size=5140,
            #num_attention_heads=60,
            #num_hidden_lkayers=40,
        )
    #print("number of parameters in model: ", model.num_parameters())
    #print("deep_size: ", objsize.get_deep_size(model)) #  calculates the "real" size (also known as "deep" size)
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = dataset["train"].column_names
    else:
        column_names = dataset["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

    logger.info("***** Tokenize dataset *****")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )
    # for GPT2
    if data_args.block_size is None: # default here
        block_size = tokenizer.model_max_length
        if block_size > config.max_position_embeddings:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    logger.info("***** Build dataset *****")
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    if training_args.do_train: # default: F
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"] # T5의 tokenizers["train"]과 같음
        if data_args.max_train_samples is not None:
            max_train_samples = min(1966029, data_args.max_train_samples)#len(train_dataset)
            train_dataset = train_dataset.select(range(max_train_samples))
    
    if training_args.do_eval: # default: F
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    # Enable tensorboard only on the master node
    has_tensorboard = is_tensorboard_available()
    if has_tensorboard:
        try:
            from flax.metrics.tensorboard import SummaryWriter
            summary_writer = SummaryWriter(log_dir=Path(training_args.output_dir))
        except ImportError as ie:
            has_tensorboard = False
            logger.warning(
                f"Unable to display metrics through TensorBoard because some package are not installed: {ie}"
            )
    else:
        logger.warning(
            "Unable to display metrics through TensorBoard because the package is not installed: "
            "Please run pip install tensorboard to enable."
        )'''

    # Initialize our training
    rng = jax.random.PRNGKey(training_args.seed)
    rng, dropout_rng = jax.random.split(rng)

    # Store some constant
    num_epochs = int(training_args.num_train_epochs)
    num_devices = alpa.get_global_num_devices()
    train_min_batch_size = (num_devices // training_args.operator_parallel //
                            training_args.pipeline_parallel * training_args.num_micro_batches)
    train_batch_size = int(training_args.per_device_train_batch_size) * num_devices
    #eval_batch_size = int(training_args.per_device_eval_batch_size) * alpa.get_global_num_devices()
    steps_per_epoch = 1966029#len(train_dataset) // train_batch_size
    total_train_steps = steps_per_epoch * num_epochs

    batch_size = train_batch_size

    print("trainig_args.mod: ", training_args.mod)
    seq_len = training_args.seq_len
    print(f"[WARNING] Please, check seq_len: {seq_len}")
    tie_word_embeddings = False
    dtype = jnp.float16
    
    batch = {
        "input_ids": jnp.ones((batch_size, seq_len), dtype=dtype),
        "attention_mask": jnp.ones((batch_size, seq_len), dtype=dtype),
        #"token_type_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32), # GPT2 X
        "position_ids": jnp.ones((batch_size, seq_len), dtype=dtype),
        "labels": jnp.ones((batch_size, seq_len), dtype=dtype),
    }
    # ---------------Alpa setting off---------------
    add_manual_remat = False # False
    add_manual_layer_marker = False
    num_manual_pipeline_stages = False
    # ----------------------------------------------

    if model_args.config_name: #"./norwegian-gpt2"
        config = AutoConfig.from_pretrained(
            model_args.config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            #n_ctx= 1024,
            #n_embd= 768,
            #n_head= 12,
            #n_layer= 12,
            #n_positions= 1024,
            #n_layer = 48,
            #vocab_size=vocab_size,
            #hidden_size=1600,
            #num_attention_heads=96,
            #intermediate_size=6400,
            #type_vocab_size=0,
            #hidden_size=5120,
            #num_attention_heads=25,
            #num_hidden_layers=40,
        )
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            #n_ctx= 1024,
            #n_embd= 768,
            #n_head= 12,
            #n_layer= 12,
            #n_positions= 1024,
            #n_layer = 48,
            #vocab_size=vocab_size,
            #hidden_size=1600,
            #num_attention_heads=96,
            #intermediate_size=6400,
            #type_vocab_size=0,
            #hidden_size=5120,
            #num_attention_heads=25,
            #num_hidden_layers=40,
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")



    print("get pretrain_model")
    if model_args.model_name_or_path: # pre-trained model 들고와
        model = FlaxAutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            seed=training_args.seed,
            dtype=jnp.float16, #getattr(jnp, model_args.dtype), # float16
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else: # 여기 들어옴.
        print("no model_name_or_path")
        model = FlaxAutoModelForCausalLM.from_config(
            config,
            seed=training_args.seed,
            dtype=jnp.float16,
            #hidden_size=5140,
            #num_attention_heads=60,
            #num_hidden_lkayers=40,
        )

    # Create learning rate schedule
    linear_decay_lr_schedule_fn = create_learning_rate_fn(
        1966029, #len(train_dataset),
        train_batch_size,
        training_args.num_train_epochs,
        training_args.warmup_steps,
        training_args.learning_rate,
    )

    # We use Optax's "masking" functionality to not apply weight decay
    # to bias and LayerNorm scale parameters. decay_mask_fn returns a
    # mask boolean with the same structure as the parameters.
    # The mask is True for parameters that should be decayed.
    # Note that this mask is specifically adapted for FlaxGPT2.
    # For other models, one should correct the layer norm parameter naming
    # accordingly.
    def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        flat_mask = {
            path: (path[-1] != "bias" and path[-2:] not in [("ln_1", "scale"), ("ln_2", "scale"), ("ln_f", "scale")])
            for path in flat_params
        }
        return traverse_util.unflatten_dict(flat_mask)

    # create adam optimizer
    if training_args.adafactor:
        # We use the default parameters here to initialize adafactor,
        # For more details about the parameters please check https://github.com/deepmind/optax/blob/ed02befef9bf81cbbf236be3d2b0e032e9ed4a40/optax/_src/alias.py#L74
        optimizer = optax.adafactor(
            learning_rate=linear_decay_lr_schedule_fn,
        )
    else:
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                learning_rate=linear_decay_lr_schedule_fn,
                b1=training_args.adam_beta1,
                b2=training_args.adam_beta2,
                eps=training_args.adam_epsilon,
                weight_decay=training_args.weight_decay,
                mask=decay_mask_fn)
        )

    # Setup train state
    if model_args.dtype == "float16":
        use_master_copy = True
        dynamic_scale = DynamicScale()
        # Fix a bug in huggingface's implementation (https://github.com/huggingface/transformers/pull/18462)
        alpa.global_config.flax_always_use_fp16_embedding = True
    else:
        use_master_copy = dynamic_scale = None
    '''params = model.init_dummy(rng, batch["input_ids"],
                              batch["attention_mask"], batch["token_type_ids"],
                              batch["position_ids"])'''
    #alpa.model.TrainState
    state = TrainState.create(apply_fn=model.__call__, params=model.params, tx=optimizer,
                              dynamic_scale=dynamic_scale, use_master_copy=use_master_copy)

    param_count = sum(x.size for x in jax.tree_leaves(model.params)) / 2**30 # GiB
    print("!!!!!!!!!!!!!!!!!!!Model parameters: ", param_count)

    def loss_fn(logits, labels):
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        loss = optax.softmax_cross_entropy(shift_logits, onehot(shift_labels, shift_logits.shape[-1]))
        return loss.mean()

    # Define gradient update step fn
    def train_step(state, batch):

        def compute_loss(params):
            labels = batch.pop("labels")
            logits = state.apply_fn(**batch, params=params, train=False)[0]
            loss = loss_fn(logits, labels)
            return loss

        dynamic_scale = state.dynamic_scale
        if dynamic_scale:
            grad_fn = dynamic_scale.value_and_grad(compute_loss)
            dynamic_scale, is_fin, loss, grads = grad_fn(state.params)
        else:
            grad_fn = alpa.value_and_grad(compute_loss)
            loss, grads = grad_fn(state.params)

        new_state = state.apply_gradients(grads=grads)

        if dynamic_scale:
            new_state = new_state.replace(
                opt_state=jax.tree_map(
                    functools.partial(jnp.where, is_fin),
                    new_state.opt_state, state.opt_state),
                params=jax.tree_map(
                    functools.partial(jnp.where, is_fin),
                    new_state.params, state.params),
                master_copy=jax.tree_map(
                    functools.partial(jnp.where, is_fin),
                    new_state.master_copy, state.master_copy),
                dynamic_scale=dynamic_scale)

        metrics = {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)}

        return new_state, metrics

    # Define eval fn
    '''def eval_step(params, batch):
        labels = batch.pop("labels")
        logits = model(**batch, params=params, train=False)[0]
        loss = loss_fn(logits, labels)
        # summarize metrics
        metrics = {"loss": loss}
        return metrics'''

    # Create parallel version of the train and eval step
    method = alpa.get_3d_parallel_method( # parallel_method.py
            num_micro_batches=training_args.num_micro_batches,
            data_parallel=-1,
            operator_parallel=training_args.operator_parallel,
            pipeline_parallel=training_args.pipeline_parallel,
            layer_num=2) # 임의의 숫자
    p_train_step = alpa.parallelize(train_step,
                                    method=method,
                                    donate_argnums=(0,))
    #p_eval_step = alpa.parallelize(eval_step) # shardparallel로 돌리는 건가?

    #min_batch_size = alpa.get_global_num_devices() * training_args.num_micro_batches # default: 4 * 4
    dump_debug_info_train_step = dump_debug_info_eval_step = True

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = 1966029")#{len(train_dataset)}") #1966029
    logger.info(f"  Num Epochs = {num_epochs}") #1
    logger.info(f"  Batch size per device (w. accumulation) = {training_args.per_device_train_batch_size}") #96
    logger.info(f"  Global train batch size (w. parallel & distributed) = {train_batch_size}") #384
    logger.info(f"  Total optimization steps = {total_train_steps}") #5119
    logger.info(f"microbatch = {training_args.num_micro_batches}")

    train_time = 0
    train_metrics = []
    print("stpes_per_epoch: ", steps_per_epoch)
    if steps_per_epoch <= 1200:
        train_epochs = 1200 // steps_per_epoch
    else: # steps_per_epoch > 110
        train_epochs = 1 # here
        steps_per_epoch = 1200
    print("train_epochs: ", train_epochs)
    epochs = tqdm(range(train_epochs), desc="Epoch ... ", position=0)
    
    step_ct = 0
    last_time = time.time()

    epochs.write("Initial compilation. This might take some minutes...")
    #key = jax.random.PRNGKey(0)
    #variables = model.init(key, np.random.randn(1, 32, 32, 3))
    #print(parameter_overview.get_parameter_overview(model))
    #print("model.num_parameters(): ", model.num_parameters())

    #def get_total_params(module: torch.nn.Module):
    #    total_params = 0
    #    for param in module.parameters():
    #        total_params += param.numel()
    #    return total_params

    #print('Total parameters in model: {:,}'.format(get_total_params(model)))
    train_time = 0
    times_alpa = []
    for epoch in epochs:
        # ======================== Training ================================

        # Create sampling rng
        rng, input_rng = jax.random.split(rng)
        train_start = time.time()
        for step in tqdm(range(steps_per_epoch), desc="Training...", position=1, leave=False):
            #print("jnp.dtype(batch): ", jnp.dtype(batch["input_ids"][0][0])) # int64
            #print("here")
            start_Time = time.time()
            state, train_metric = p_train_step(state, batch) # 안에서 executable.profile_with_dummy_inputs()
            #alpa_executable = p_train_step.get_executable(state, batch)
            #print(f"Alpa execution per GPU memory usage:   {alpa_executable.get_total_allocation_size() / GB:.2f} GB")
            final_Time = time.time()-start_Time
            times_alpa.append(final_Time)
            print("final_time: ", final_Time)
            #_loss = train_metric["loss"]
            #print(f"loss: {_loss}")
            #print("end")
            train_time += time.time() - train_start
            train_metrics.append(train_metric)

            cur_step = epoch * (1966029 // train_batch_size) + step #len(train_dataset) // train_batch_size) + step
            #if has_tensorboard:
            #    write_train_metric(summary_writer, train_metrics, train_time, cur_step)
    #times_alpa.pop(0)
    print(f"total execution time: {np.sum(times_alpa[1:])} [s] without warmup")
    print(f"Serial execution time. Mean: {round(np.mean(times_alpa), 5):.5f} s, Std: {round(np.std(times_alpa), 5):.5f} s")
    print(f"Serial execution time. Mean: {round(np.mean(times_alpa[200:]), 5):.5f} s, Std: {round(np.std(times_alpa[200:]), 5):.5f} s")


if __name__ == "__main__":
    main()