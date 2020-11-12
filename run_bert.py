# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
get_linear_schedule_with_warmup
)

from transformers import glue_convert_examples_to_features as convert_examples_to_features

from bert_processor import active_processors as processors
from bert_processor import active_output_modes as output_modes
from bert_processor import active_lang_dic as lang_dic

from bert_config import BertInitConfig

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

import sys
import argparse
import pickle
from datetime import datetime
import logging
from os.path import join
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import os

import strategy

logger = logging.getLogger(__name__)

# ALL_MODELS = sum(
#     (
#         tuple(conf.pretrained_config_archive_map.keys())
#         for conf in (BertConfig, XLNetConfig, XLMConfig, RobertaConfig, DistilBertConfig)
#     ),
#     (),
# )

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer)
}

def set_seed(args):
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    # if args.local_rank in [-1, 0]:
    #     tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    print('args.train_batch_size', args.train_batch_size)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        print('load optimizer and scheduler from pretrained_file')
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    print("***** Running training *****")
    print("  Num examples = ", len(train_dataset))
    print("  Num Epochs = ", args.num_train_epochs)
    # print("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    # print(
    #     "  Total train batch size (w. parallel, distributed & accumulation) = %d",
    #     args.train_batch_size
    #     * args.gradient_accumulation_steps
    #     * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    # )
    # print("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    # print("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        # gobal_step 12

        # print ('model_name_or_path', args.model_name_or_path)
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        # print ('gobal_step', global_step)
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        # print("  Continuing training from checkpoint, will skip to saved global_step")
        # print("  Continuing training from epoch %d", epochs_trained)
        # print("  Continuing training from global step %d", global_step)
        # print("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility
    for _ in train_iterator:
        # epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        epoch_iterator = train_dataloader
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet"] else None
                )  # XLM, DistilBERT and RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                '''
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))


                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    print("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    print("Saving optimizer and scheduler states to %s", output_dir)
                '''

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    # if args.local_rank in [-1, 0]:
    #     tb_writer.close()

    return global_step, tr_loss / global_step

def evaluate(args, model, tokenizer, processor, prefix="", set_type='test'):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_examples, eval_features = \
            load_and_cache_features(args, eval_task, tokenizer, processor, set_type=set_type)
        eval_dataset = convert_features_to_dataset(eval_features)

        # if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        #     os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        print("***** Running evaluation {} *****".format(prefix))
        print("  Num examples = ", len(eval_dataset))
        print("  Batch size = ", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        # for batch in tqdm(eval_dataloader, desc="Evaluating"):
        for batch in eval_dataloader:
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1],
                          "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet"] else None
                    )  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)

        y_pred = preds
        y_test = out_label_ids
        micro_f1 = f1_score(y_test, y_pred, average='micro')
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average=None)
        indexes = np.argsort(f1)
        # accuracies = np.sort(accuracies)
        matrix = confusion_matrix(y_test, y_pred)

        prec = precision_score(y_test, y_pred, average=None)
        recall = recall_score(y_test, y_pred, average=None)

    return matrix, indexes, micro_f1, macro_f1, list(f1), list(prec), list(recall)


def load_and_cache_features(args, task, tokenizer, processor, set_type='test'):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_features_{}_{}_{}_{}_{}".format(
            set_type,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
            str(processor.num_classes)
        ),
    )
    cached_examples_file = os.path.join(
        args.data_dir,
        "cached_examples_{}_{}_{}_{}_{}".format(
            set_type,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
            str(processor.num_classes)
        ),
    )
    print('cached_features_file', cached_features_file)
    if os.path.exists(cached_features_file) and \
            os.path.exists(cached_examples_file) and \
            not args.overwrite_cache:
        print("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        examples = torch.load(cached_examples_file)
    else:
        print("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        if set_type == 'dev':
            examples = processor.get_dev_examples()
        elif set_type == 'train':
            examples = processor.get_train_examples()
        elif set_type == 'test':
            examples = processor.get_test_examples()

        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )
        features = np.array(features)
        examples = np.array(examples)
        if args.local_rank in [-1, 0]:
            print("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
            torch.save(examples, cached_examples_file)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    return examples, features


def convert_features_to_dataset(features, output_mode='classification'):
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def init_model(task_name, processor):
    args = BertInitConfig()

    print('args.per_gpu_train_batch_size', args.per_gpu_train_batch_size)
    args.per_gpu_train_batch_size *= 2

    args.num_train_epochs = 5

    args.task_name = task_name
    args.model_name_or_path = lang_dic[args.task_name]
    args.data_dir = join('./bert_cached_{}'.format(args.task_name))
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    args.output_dir = join('./tmp', args.task_name)

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name
    # if args.task_name not in processors:
    #     raise ValueError("Task not found: %s" % (args.task_name))
    # processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        output_hidden_states=True,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    print('config.output_hidden_states', config.output_hidden_states)
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    # print("Training/evaluation parameters %s", args)

    return args, model, tokenizer


def predict(args, model, eval_features):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = convert_features_to_dataset(eval_features)

        # if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        #     os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                     batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        print("***** Running prediction *****")
        print("  Num examples = ", len(eval_dataset))
        print("  Batch size = ", args.eval_batch_size)
        probs = None
        sentence_vectors = None
        # for batch in tqdm(eval_dataloader, desc="Predicing"):
        for batch in eval_dataloader:
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1],
                          "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet"] else None
                    )  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits, hidden_states = outputs

                logits = nn.Softmax(dim=1)(logits)
                last_layer_hidden_states = hidden_states[-1]
                sen_emb = torch.mean(last_layer_hidden_states, dim=1)

            if probs is None:
                probs = logits.detach().cpu().numpy()
                sentence_vectors = sen_emb.detach().cpu().numpy()
            else:
                probs = np.append(probs, logits.detach().cpu().numpy(), axis=0)
                sentence_vectors = np.append(sentence_vectors, sen_emb.detach().cpu().numpy(), axis=0)

    return probs, sentence_vectors


def inverse_transform(label_map_inverse, idx_list):
    inverse_idx_list = []
    for idx in idx_list:
        inverse_idx_list.append(label_map_inverse[idx])
    return inverse_idx_list


def generate(examples, file):
    with open(file, 'wb') as f:
        pickle.dump(examples, f)

def get_label_from_feature(features, label_map_inverse):
    label_list, label_list_inverse = [], []
    for f in features:
        label = f.label
        label_inverse = label_map_inverse[label]
        label_list.append(label)
        label_list_inverse.append(label_inverse)
    return label_list, label_list_inverse

def get_distribution_dic(features, label_map_inverse):
    distribution_dic = {}
    distribution_dic_inverse = {}
    for i in label_map_inverse:
        label = label_map_inverse[i]
        distribution_dic[i] = 0
        distribution_dic_inverse[label] = 0
    label_list, label_list_inverse = get_label_from_feature(features, label_map_inverse)
    for i in range(len(label_list)):
        label = label_list[i]
        label_inverse = label_list_inverse[i]
        distribution_dic[label] += 1
        distribution_dic_inverse[label_inverse] += 1
    return distribution_dic, distribution_dic_inverse

def train_all(epoch, input_args,
              args, model, tokenizer, processor,
              train_examples, train_features,
              initial_batch, step,
              number=2, mode=strategy.MODE_RANDOM, factor_mode=strategy.FACTOR_MODE_BASE):
    initial_time = datetime.now()
    times = []

    micro_f1_list, macro_f1_list, f1_list, prec_list, recall_list = [], [], [], [], []

    dev_micro_f1_list, dev_macro_f1_list, dev_f1_list, dev_prec_list, dev_recall_list = [], [], [], [], []

    batch_size_list = [initial_batch]
    label_map_inverse = processor.get_label_map_inverse()

    batch_features = train_features
    batch_examples = train_examples

    logging.info('dataset:{}, num_classes:{}, mode:{}, factor_mode:{}, '.format(
        input_args.dataset, input_args.number, input_args.mode, input_args.factor_mode) +
                 "Trained on dataset:" + str(batch_features.shape[0]))

    train_dataset = convert_features_to_dataset(batch_features)
    global_step, tr_loss = train(args, train_dataset, model, tokenizer)
    print(" global_step = %s, average loss = %s", global_step, tr_loss)

    distribution_dic, distribution_dic_inverse = get_distribution_dic(batch_features, label_map_inverse)

    matrix, indexes, micro_f1, macro_f1, f1, prec, recall = \
        evaluate(args, model, tokenizer, processor, prefix="", set_type='test')

    tmp_sorted_accr = np.sort(f1)
    bottom_labels = inverse_transform(label_map_inverse, indexes[:10])
    top_labels = inverse_transform(label_map_inverse, indexes[-10:])
    print("Least ten classes and accuracies")
    print(list(zip(bottom_labels, tmp_sorted_accr[:10])))
    print("Top ten classes and accuracies")
    print(list(zip(top_labels[::-1], tmp_sorted_accr[-10:][::-1])))
    print("Micro: %.4f, Macro: %.4f" % (micro_f1, macro_f1))
    print()
    micro_f1_list.append(micro_f1)
    macro_f1_list.append(macro_f1)
    f1_list.append(f1)
    prec_list.append(prec)
    recall_list.append(recall)

    time = datetime.now()
    times.append((time - initial_time).seconds)

    return batch_size_list, micro_f1_list, macro_f1_list, f1_list, prec_list, recall_list, \
           dev_micro_f1_list, dev_macro_f1_list, dev_f1_list, dev_prec_list, dev_recall_list

def test_all(epoch, input_args,
             args, model, tokenizer, processor,
             train_examples, train_features,
             initial_batch, step,
             number=2, mode=strategy.MODE_RANDOM, factor_mode=strategy.FACTOR_MODE_BASE):
    times = []

    micro_f1_list, macro_f1_list, f1_list, prec_list, recall_list = [], [], [], [], []

    dev_micro_f1_list, dev_macro_f1_list, dev_f1_list, dev_prec_list, dev_recall_list = [], [], [], [], []

    batch_size_list = [initial_batch]
    label_map_inverse = processor.get_label_map_inverse()
    print('label_map_inverse', label_map_inverse)

    batch_features = train_features[:initial_batch]#training data
    batch_examples = train_examples[:initial_batch]

    data_features = train_features[initial_batch:]#candidate data
    data_examples = train_examples[initial_batch:]


    logging.info('dataset:{}, num_classes:{}, mode:{}, factor_mode:{}, '.format(
        input_args.dataset, input_args.number, input_args.mode, input_args.factor_mode) +
                 "Trained on dataset:" + str(batch_features.shape[0]))

    distribution_dic, distribution_dic_inverse = get_distribution_dic(batch_features, label_map_inverse)

    train_dataset = convert_features_to_dataset(batch_features)
    initial_time = datetime.now()
    global_step, tr_loss = train(args, train_dataset, model, tokenizer)
    time = datetime.now()
    times.append((time - initial_time).seconds)

    print(" global_step = %s, average loss = %s", global_step, tr_loss)

    matrix, indexes, micro_f1, macro_f1, f1, prec, recall = \
        evaluate(args, model, tokenizer, processor, prefix="", set_type='test')


    tmp_sorted_accr = np.sort(f1)
    bottom_labels = inverse_transform(label_map_inverse, indexes[:10])
    top_labels = inverse_transform(label_map_inverse, indexes[-10:])
    all_labels = inverse_transform(label_map_inverse, indexes)
    print("Least ten classes and accuracies")
    print(list(zip(bottom_labels, tmp_sorted_accr[:10])))
    print("Top ten classes and accuracies")
    print(list(zip(top_labels[::-1], tmp_sorted_accr[-10:][::-1])))
    print("All classes and accuracies")
    print(list(zip(all_labels, tmp_sorted_accr)))
    print("Micro: %.8f, Macro: %.8f" % (micro_f1, macro_f1))
    print()
    micro_f1_list.append(micro_f1)
    macro_f1_list.append(macro_f1)
    f1_list.append(f1)
    prec_list.append(prec)
    recall_list.append(recall)

    while data_features.shape[0]:

        if batch_features.shape[0] >= 3000:
            break

        candidate_index = np.random.choice(data_features.shape[0], min(data_features.shape[0], step*10), replace=False)
        candidate = data_features[candidate_index]
        candidate_predictions, candidate_vectors = predict(args, model, candidate)

        _, train_vectors = predict(args, model, batch_features)
        train_labels, _ = get_label_from_feature(batch_features, label_map_inverse)

        if mode == strategy.MODE_RANDOM:
            best = strategy.select_random(step, data_features.shape[0])
        elif mode == strategy.MODE_ENTROPY:
            best = strategy.select_entropy(candidate_predictions,
                                                     distribution_dic,
                                                     step=step, factor_mode=factor_mode)
        elif mode == strategy.MODE_PURITY:
            best = strategy.select_purity(candidate_predictions, candidate_vectors,
                                           distribution_dic, step=step, factor_mode=factor_mode)
        elif mode == strategy.MODE_ACTIVE:
            best = strategy.select_active(candidate_predictions,
                                                    distribution_dic, step=step, factor_mode=factor_mode)
        elif mode == strategy.MODE_CENTER:
            best = strategy.select_center(candidate_predictions, candidate_vectors,
                                                    train_labels, train_vectors,
                                                    distribution_dic, step=step, factor_mode=factor_mode)
        elif mode == strategy.MODE_RADIUS_MULTI_LABEL_UN_CENTROID:
            best = strategy.select_radius_multi_label_unlabel_centroid(
                                            candidate_predictions, candidate_vectors,
                                            train_labels, train_vectors,
                                           distribution_dic, step=step, factor_mode=factor_mode)

        original_index = candidate_index[best]

        batch_features = np.concatenate((batch_features, data_features[original_index]))
        batch_examples = np.concatenate((batch_examples, data_examples[original_index]))
        data_features = np.delete(data_features, original_index, 0)
        data_examples = np.delete(data_examples, original_index, 0)

        print("Pool size:" + str(batch_features.shape))
        batch_size_list.append(batch_features.shape[0])

        distribution_dic, distribution_dic_inverse = get_distribution_dic(batch_features, label_map_inverse)

        del model, tokenizer
        args, model, tokenizer = init_model(args.task_name, processor)
        train_dataset = convert_features_to_dataset(batch_features)

        initial_time = datetime.now()
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        time = datetime.now()
        times.append((time - initial_time).seconds)
        print('train time: {}'.format((time - initial_time).seconds))

        matrix, indexes, micro_f1, macro_f1, f1, prec, recall = \
            evaluate(args, model, tokenizer, processor, prefix="", set_type='test')

        tmp_sorted_accr = np.sort(f1)
        bottom_labels = inverse_transform(label_map_inverse, indexes[:10])
        top_labels = inverse_transform(label_map_inverse, indexes[-10:])
        print("Least ten classes and accuracies")
        print(list(zip(bottom_labels, tmp_sorted_accr[:10])))
        print("Top ten classes and accuracies")
        print(list(zip(top_labels[::-1], tmp_sorted_accr[-10:][::-1])))
        print("All classes and accuracies")
        print(list(zip(all_labels, tmp_sorted_accr)))
        print("Micro: %.8f, Macro: %.8f" % (micro_f1, macro_f1))
        print('\n')
        micro_f1_list.append(micro_f1)
        macro_f1_list.append(macro_f1)
        f1_list.append(f1)
        prec_list.append(prec)
        recall_list.append(recall)

    print('test_all, time:', time)
    logging.info('test_all, time:', time)
    return batch_size_list, micro_f1_list, macro_f1_list, f1_list, prec_list, recall_list, \
           dev_micro_f1_list, dev_macro_f1_list, dev_f1_list, dev_prec_list, dev_recall_list, \
            times

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='tnews',
                        type=str, help="")

    parser.add_argument('-n', '--number', default=2,
                        type=int, help="2,5,10,-1(all)")

    parser.add_argument('-m', '--mode', default=strategy.MODE_RANDOM,
                        type=int, help="")
    parser.add_argument('-fm', '--factor_mode', default=strategy.FACTOR_MODE_BASE,
                        type=int, help="")

    parser.add_argument('-ib', '--initial_batch', default=100,
                        type=int, help="")
    parser.add_argument('-val', '--validation', default=0,
                        type=int, help="")

    if len(sys.argv) == 0:
        parser.print_help()
        sys.exit(1)

    print("")
    args = parser.parse_args()
    for arg in vars(args):
        print("{}={}".format(arg.upper(), getattr(args, arg)))
    print("")

    return args


if __name__ == '__main__':
    input_args = parse_args()

    mode = input_args.mode
    factor_mode = input_args.factor_mode
    num_classes = input_args.number
    initial_batch = input_args.initial_batch
    task_name = input_args.dataset
    val = input_args.validation

    # Training
    processor = processors[task_name](num_classes=num_classes)
    args, model, tokenizer = init_model(task_name, processor)

    train_examples, train_features = \
        load_and_cache_features(args, task_name, tokenizer, processor, set_type='train')

    print('train_examples', train_examples[:10])

    batch_size, micro_f1, macro_f1, f1, prec, recall = [], [], [], [], [], []
    dev_micro_f1, dev_macro_f1, dev_f1, dev_prec, dev_recall = [], [], [], [], []
    times = []

    for i in range(3):
        print('epoch: ', i)
        args, model, tokenizer = init_model(task_name, processor)

        if initial_batch == -1:
            batch_size_list, micro_f1_list, macro_f1_list, f1_list, prec_list, recall_list, \
            dev_micro_f1_list, dev_macro_f1_list, dev_f1_list, dev_prec_list, dev_recall_list = \
                train_all(i, input_args,
                          args, model, tokenizer, processor,
                          train_examples, train_features,
                          initial_batch, step=100,
                          number=num_classes, mode=mode, factor_mode=factor_mode)
        else:
            batch_size_list, micro_f1_list, macro_f1_list, f1_list, prec_list, recall_list, \
            dev_micro_f1_list, dev_macro_f1_list, dev_f1_list, dev_prec_list, dev_recall_list, times_list = \
                test_all(i, input_args,
                         args, model, tokenizer, processor,
                         train_examples, train_features,
                         initial_batch, step=100,
                         number=num_classes, mode=mode, factor_mode=factor_mode)

        batch_size.append(batch_size_list)
        micro_f1.append(micro_f1_list)
        macro_f1.append(macro_f1_list)
        f1.append(f1_list)
        prec.append(prec_list)
        recall.append(recall_list)

        dev_micro_f1.append(dev_micro_f1_list)
        dev_macro_f1.append(dev_macro_f1_list)
        dev_f1.append(dev_f1_list)
        dev_prec.append(dev_prec_list)
        dev_recall.append(dev_recall_list)
        times.append(times_list)
        print('epoch:{} macro:{}'.format(i, str(macro_f1_list)))
        if initial_batch == -1:
            break

    out_dir = 'res/bert_{}'.format(task_name)
    if initial_batch == -1:
        out_dir += '_all'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(join(out_dir, '{}@{}@{}@{}.txt'.format(
            num_classes, mode, factor_mode, initial_batch)), 'w') as f:

        f.write(str(batch_size) + '\n')
        f.write(str(micro_f1) + '\n')
        f.write(str(macro_f1) + '\n')
        f.write(str(prec) + '\n')
        f.write(str(recall) + '\n')
        f.write(str(f1) + '\n')

        f.write(str(dev_micro_f1) + '\n')
        f.write(str(dev_macro_f1) + '\n')
        f.write(str(dev_prec) + '\n')
        f.write(str(dev_recall) + '\n')
        f.write(str(dev_f1) + '\n')
        f.write(str(times) + '\n')
        f.flush()
    print('DONE')

    out_list = ['task_name', task_name,
                'num_classes', num_classes,
                'mode', mode,
                'factor_mode', factor_mode,
                'initial_batch', initial_batch]
    logging.info("DONE\t" + ','.join(str(i) for i in out_list))
