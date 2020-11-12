# encoding=utf-8
import fasttext as ft
import sys
import numpy as np

import argparse
import codecs

import logging
import random

from os import listdir
from os.path import join

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import os

from processor import Processor, MODE_SPLITTED, MODE_MIXED

import strategy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(funcName)s - %(message)s')

def evaluate(model, x_test, y_test):
    y_pred = []

    for l in x_test:
        x_pred = model.predict(l)
        pred = int(x_pred[0][0].split("__")[-1])
        y_pred.append(pred)

    y_pred = np.array(y_pred)
    y_test = y_test.argmax(axis=1)
    accuracy = f1_score(y_test, y_pred, average='micro')
    accuracy2 = f1_score(y_test, y_pred, average='macro')

    accuracies = f1_score(y_test, y_pred, average=None)
    indexes = np.argsort(accuracies)
    accuracies = np.sort(accuracies)

    prec = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)

    matrix = confusion_matrix(y_test, y_pred)
    return accuracy, matrix, accuracy2, indexes, list(accuracies), list(prec), list(recall)

def predict(model, data, k):
    predictions = []
    for l in data:
        indexes, figures = model.predict(l, k)
        x = np.zeros(k)
        for i in range(len(indexes)):
            ind = int(indexes[i].split("__")[-1])
            x[ind] = figures[i]
        predictions.append(x)
    predictions = np.array(predictions)
    return predictions

def get_sentence_vector(model, data):
    sentence_vectors = []
    for line in data:
        sentence_vector = model.get_sentence_vector(line)
        sentence_vectors.append(sentence_vector)
    sentence_vectors = np.array(sentence_vectors)
    return sentence_vectors

def generate(data, label, file):
    with codecs.open(file, 'w', encoding='utf8') as f:
        for i, text in enumerate(data):
            l = label[i].argmax()
            f.write("__label__" + str(l) + '\t' + text + '\n')

def fetch_data_by_index(data, index_list):
    new_data = []
    for index in index_list:
        new_data.append(data[index])
    return new_data

def del_data_by_index(data, index_list):
    new_data = []
    for i in range(len(data)):
        if i in index_list:
            continue
        new_data.append(data[i])
    return new_data

def get_label_from_feature(batch_y):
    batch_y_inverse = batch_y.argmax(axis=1)
    return batch_y_inverse

def test_all(epoch, x_train, y_train, x_test, y_test,
             initial_batch, step, k, encoder, fe, lr,
             dataset, num_classes,
             number=2, mode=strategy.MODE_RANDOM,
             factor_mode=strategy.FACTOR_MODE_BASE, norm_mode=strategy.NORM_MODE_MAX,
             loss='softmax', pretrainedVectors='../small.300.vec', dim=300,
             debug=0, use_sigmoid=1):
    accuracies = []
    macro_accr = []
    f1_list = []
    prec_list, recall_list = [], []

    l = [initial_batch]

    out_dir = 'ft_sample_{}'.format(dataset)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    batch_x, batch_y = x_train[0:initial_batch], y_train[0:initial_batch]
    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)

    f = join(out_dir, '{}_{}_{}_{}_{}_{}.txt'.format(
        number, mode, factor_mode, epoch, batch_y.shape[0], fe))
    generate(batch_x, batch_y, f)

    data, label = x_train[initial_batch:], y_train[initial_batch:]
    shuffled_index = list(range(len(data)))
    random.shuffle(shuffled_index)
    data = fetch_data_by_index(data, shuffled_index)
    label = fetch_data_by_index(label, shuffled_index)

    model = ft.train_supervised(f, lr=lr, epoch=fe, wordNgrams=2, dim=dim, loss=loss,
                                minn=2, maxn=3, pretrainedVectors=pretrainedVectors, verbose=0)

    batch_y_inverse = batch_y.argmax(axis=1)
    distribution_dic = {}
    for idx in range(batch_y.shape[1]):
        distribution_dic[idx] = 0
    for labeled_y in batch_y_inverse:
        distribution_dic[labeled_y] += 1

    accr, _, accr2, indexes, sorted_accr, prec, recall = evaluate(model, x_test, y_test)

    bottom_labels = encoder.inverse_transform(indexes[:10])
    top_labels = encoder.inverse_transform(indexes[-10:])
    all_labels = encoder.inverse_transform(indexes)
    print("Least ten classes and accuracies")
    print(list(zip(bottom_labels, sorted_accr[:10])))
    print("Top ten classes and accuracies")
    print(list(zip(top_labels[::-1], sorted_accr[-10:][::-1])))

    print("all classes and accuracies")
    print(list(zip(all_labels[::-1], sorted_accr[::-1])))
    print("Micro: %.4f, Macro: %.4f" % (accr, accr2))
    print()
    accuracies.append(accr)
    macro_accr.append(accr2)
    f1_list.append(sorted_accr)
    prec_list.append(prec)
    recall_list.append(recall)

    while len(data):

        if batch_x.shape[0] >= 3000:
            break

        candidate_index = np.random.choice(data.shape[0], min(data.shape[0], step*10), replace=False)
        candidate = data[candidate_index]
        candidate_predictions = predict(model, candidate, num_classes)
        candidate_vectors = get_sentence_vector(model, candidate)

        train_vectors = get_sentence_vector(model, batch_x)
        train_labels = get_label_from_feature(batch_y)

        if mode == strategy.MODE_RANDOM:
            best = strategy.select_random(step, len(data))
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
        tmp_x = fetch_data_by_index(data, original_index)
        tmp_y = fetch_data_by_index(label, original_index)

        batch_x = np.concatenate((batch_x, np.array(tmp_x)))
        batch_y = np.vstack((batch_y, np.array(tmp_y)))

        data = del_data_by_index(data, original_index)
        label = del_data_by_index(label, original_index)

        batch_x, batch_y = shuffle(batch_x, batch_y)

        f = join(out_dir, '{}_{}_{}_{}_{}_{}.txt'.format(
            number, mode, factor_mode, epoch, batch_y.shape[0], fe))
        generate(batch_x, batch_y, f)

        l.append(batch_x.shape[0])

        del model
        model = ft.train_supervised(f, lr=lr, epoch=fe, wordNgrams=2, dim=dim, loss=loss,
                                    minn=2, maxn=3, pretrainedVectors=pretrainedVectors, verbose=0)

        batch_y_inverse = batch_y.argmax(axis=1)
        distribution_dic = {}
        for idx in range(batch_y.shape[1]):
            distribution_dic[idx] = 0
        for labeled_y in batch_y_inverse:
            distribution_dic[labeled_y] += 1

        accr, _, accr2, indexes, sorted_accr, prec, recall = evaluate(model, x_test, y_test)

        bottom_labels = encoder.inverse_transform(indexes[:10])
        top_labels = encoder.inverse_transform(indexes[-10:])
        all_labels = encoder.inverse_transform(indexes)
        print("Least ten classes and accuracies")
        print(list(zip(bottom_labels, sorted_accr[:10])))
        print("Top ten classes and accuracies")
        print(list(zip(top_labels[::-1], sorted_accr[-10:][::-1])))

        print("all classes and accuracies")
        print(list(zip(all_labels[::-1], sorted_accr[::-1])))
        print("Micro: %.4f, Macro: %.4f" % (accr, accr2))
        print()
        accuracies.append(accr)
        macro_accr.append(accr2)
        f1_list.append(sorted_accr)
        prec_list.append(prec)
        recall_list.append(recall)
    return l, accuracies, macro_accr, f1_list, prec_list, recall_list


def get_emb_dim(dataset):
    if dataset in ['yanjing_tokenized', 'tnews_tokenized', 'book_tokenized']:
        return '../small.300.vec', 300
    in_dir = join('../data', dataset)
    files = [f for f in listdir(in_dir) if f.endswith('.emb')]
    if len(files) != 1:
        print('not single one emb file!', dataset, files, flush=True)
        sys.exit()
    emb_file = join(in_dir, files[0])
    f = open(emb_file, 'r')
    line = f.readline().strip().split()
    dim = int(line[1])
    return emb_file, dim


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='tnews_tokenized', required=True,
                        type=str, help="")

    parser.add_argument('-n', '--number', default=-1, required=True,
                        type=int, help="2,5,10,-1(all)")

    parser.add_argument('-m', '--mode', default=strategy.MODE_RANDOM, required=True,
                        type=int, help="")
    parser.add_argument('-fm', '--factor_mode', default=strategy.FACTOR_MODE_FREQ,
                        type=int, help="")
    parser.add_argument('-nm', '--norm_mode', default=strategy.NORM_MODE_MINMAX,
                        type=int, help="")
    parser.add_argument('-s', '--use_sigmoid', default=1,
                        type=int, help="")

    parser.add_argument('-ib', '--initial_batch', default=100,
                        type=int, help="")
    parser.add_argument('-fe', '--fasttext_epoch', default=25,
                        type=int, help="")
    parser.add_argument('-val', '--validation', default=0,
                        type=int, help="")
    parser.add_argument('--debug', default=1,
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


def main():
    args = parse_args()

    mode = args.mode
    factor_mode = args.factor_mode
    norm_mode = args.norm_mode
    number = args.number
    initial_batch = args.initial_batch
    dataset = args.dataset
    fe = args.fasttext_epoch
    val = args.validation
    debug = args.debug
    use_sigmoid = args.use_sigmoid

    processors = {
        'reuters_tokenized': MODE_MIXED,
        'news_tokenized': MODE_MIXED,
        'SearchSnippets_tokenized': MODE_MIXED,
        'tnews_tokenized': MODE_SPLITTED,
        'yanjing_tokenized': MODE_SPLITTED,
        'book_tokenized': MODE_MIXED,
    }

    lang_dic = {
        'reuters_tokenized': 'eng',
        'news_tokenized': 'eng',
        'SearchSnippets_tokenized': 'eng',
        'tnews_tokenized': 'cn',
        'yanjing_tokenized': 'cn',
        'book_tokenized': 'cn',
    }

    if dataset not in processors or dataset not in lang_dic:
        raise ValueError("Task not found: %s" % (dataset))

    lang = lang_dic[dataset]
    processor = Processor(mode=processors[dataset], dataset=dataset, lang=lang)

    train_x, train_y, test_x, test_y, dev_x, dev_y, num_classes, encoder = \
        processor.get_data(number)

    print('train: {} test: {} dev: {}'.format(
        train_y.shape, test_y.shape, dev_y.shape if dev_y is not None else 'None'))

    # to check the metric
    # test_x = dev_x
    # test_y = dev_y

    print('num_classes', num_classes)
    logging.info('train: {}, test: {}'.format(train_y.shape, test_y.shape))

    loss = 'softmax'

    pretrainedVectors, dim = get_emb_dim(dataset)
    logging.info('dataset:{} pretrainedVectors:{} dim:{}'.format(
        dataset, pretrainedVectors, dim))

    batch_list, accuracy, time = [], [], []
    f1_all = []
    prec_all, recall_all = [], []
    for i in range(3):
        print('epoch: ', i)
        l, accr, tmp_time, f1_list, prec_list, recall_list = \
            test_all(i, train_x, train_y, test_x, test_y,
                      initial_batch, 100, 2, encoder,
                      25,
                      0.1,
                      dataset=dataset, num_classes=num_classes,
                      number=number, mode=mode,
                      factor_mode=factor_mode, norm_mode=norm_mode,
                      loss=loss,
                      pretrainedVectors=pretrainedVectors, dim=dim, debug=debug,
                      use_sigmoid=use_sigmoid)

        batch_list.append(l)
        accuracy.append(accr)
        time.append(tmp_time)
        f1_all.append(f1_list)
        prec_all.append(prec_list)
        recall_all.append(recall_list)

    out_dir = 'res/ft_{}'.format(dataset)
    if initial_batch == -1:
        out_dir += '_all'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    print('out_dir', out_dir)
    with open(join(out_dir, '{}@{}@{}@{}.txt'.format(
            num_classes, mode, factor_mode, initial_batch)), 'w') as f:
        f.write(str(batch_list) + '\n')
        f.write(str(accuracy) + '\n')  # micro
        f.write(str(time) + '\n')  # macro
        f.write(str(prec_all) + '\n')
        f.write(str(recall_all) + '\n')
        f.write(str(f1_all) + '\n')
    print('DONE')

    out_list = [dataset, num_classes, mode, factor_mode, norm_mode]
    logging.info("DONE\t" + ','.join(str(i) for i in out_list))

main()
