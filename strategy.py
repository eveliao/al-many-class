import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
from sklearn.metrics import pairwise_distances
import math

MODE_RANDOM = 0
MODE_ENTROPY = 1
MODE_PURITY = 2
MODE_ACTIVE = 3
MODE_CENTER = 4
MODE_RADIUS_MULTI_LABEL_UN_CENTROID = 5

FACTOR_MODE_BASE = 0
FACTOR_MODE_FREQ = 1
NORM_MODE_MAX = 0
NORM_MODE_MINMAX = 1

def select_random(step, shape):
    original_index = range(min(step, shape))
    return original_index

def sort_freq(distribution_dic):
    # sort freq, more samples, rank larger
    # labels with the same number of samples have the same rank
    # e.g. [1:0, 2:1, 3:1, 4:2] -> [1:1, 2:2, 3:2, 4:3]
    sorted_by_value = sorted(distribution_dic.items(), key=lambda kv: kv[1])
    print('sort_freq sorted_by_value', sorted_by_value)
    dic = {}
    rank = 1
    for i, tup in enumerate(sorted_by_value):
        if i == 0:
            dic[tup[0]] = rank
        else:
            if tup[1] != sorted_by_value[i - 1][1]:
                rank += 1
            dic[tup[0]] = rank
    print ('sort_freq dic', dic)
    return dic

def norm_list_max(array):
    array_select = [i for i in array if i != float('inf')]
    array = np.array(array)
    array = array / max(array_select)
    return array

def norm_list_min_max(array):
    min_val = min(array)
    inf_indices = []
    for i, val in enumerate(array):
        if val == float('inf'):
            inf_indices.append(i)
            array[i] = min_val
    array = np.array(array).reshape(-1, 1)
    scaler_similarity = MinMaxScaler()
    array = scaler_similarity.fit_transform(array)
    array = np.squeeze(array)
    for i in inf_indices:
        array[i] = float('inf')
    return array

def select_entropy(candidate_predictions,
                   distribution_dic,
                   step=100,
                   factor_mode=FACTOR_MODE_BASE, norm_mode=NORM_MODE_MAX,
                   debug=False):

    entropies = []
    for prediction in candidate_predictions:
        entropy = 0
        for p in prediction:
            if p:
                entropy -= p * np.log2(p)
        entropies.append(entropy)
    if norm_mode == NORM_MODE_MAX:
        entropies = norm_list_max(entropies)
    else:
        entropies = norm_list_min_max(entropies)

    rank_dic = sort_freq(distribution_dic)

    estimates = []
    for i, e in enumerate(entropies):
        predict_y = candidate_predictions[i].argmax()
        if factor_mode == FACTOR_MODE_FREQ and predict_y in rank_dic:
            estimates.append(e / rank_dic[predict_y])
        else:
            estimates.append(e)

    estimates = np.array(estimates)
    entropies_index = np.argsort(estimates)[::-1]

    index = entropies_index[:step]
    return index


def select_purity(candidate_predictions, candidate_vectors,
                  distribution_dic,
                  step=100,
                  factor_mode=FACTOR_MODE_BASE, norm_mode=NORM_MODE_MAX,
                  debug=False):

    entropies = []
    for prediction in candidate_predictions:
        entropy = 0
        for p in prediction:
            if p:
                entropy -= p * np.log2(p)
        entropies.append(entropy)
    entropies = np.array(entropies)

    rank_dic = sort_freq(distribution_dic)

    # calculate the centroid of each class
    centroid = {}
    for index in range(candidate_vectors.shape[0]):
        belongsto = candidate_predictions[index].argmax()
        if belongsto not in centroid.keys():
            centroid[belongsto] = [candidate_vectors[index]]
        else:
            centroid[belongsto].append(candidate_vectors[index])
        if debug:
            if index == 0:
                print ('belongsto', belongsto)
                print ('index: {}, centroid: {} {}'.format(index, centroid[belongsto], type(centroid[belongsto])))

    for belongsto in centroid.keys():
        centroid[belongsto] = np.array(centroid[belongsto])
        center = np.mean(centroid[belongsto], axis=0)
        centroid[belongsto] = center

    similarities = []
    for i, vector in enumerate(candidate_predictions):
        belongsto = np.argmax(vector)
        similarity = cosine_similarity([candidate_vectors[i]], [centroid[belongsto]])[0][0]
        if debug:
            if i == 0:
                print ('cos sim', cosine_similarity([candidate_vectors[i]], [centroid[belongsto]]))
                print ('similarity', similarity)
        similarities.append(similarity)

    similarities = np.array(similarities)

    estimates = []
    for i, vector in enumerate(candidate_predictions):
        estimates.append(entropies[i] * similarities[i])

    if norm_mode == NORM_MODE_MAX:
        estimates = norm_list_max(estimates)
    else:
        estimates = norm_list_min_max(estimates)

    for i, vector in enumerate(candidate_predictions):
        belongsto = np.argmax(vector)
        if factor_mode == FACTOR_MODE_FREQ and belongsto in rank_dic:
            estimates[i] /= rank_dic[belongsto]

    estimates = np.array(estimates)
    best = np.argsort(estimates)[::-1][:step]

    return best


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def select_active(candidate_predictions,
                  distribution_dic,
                  step=100, factor_mode=FACTOR_MODE_BASE, norm_mode=NORM_MODE_MAX,
                  debug=False):
    rank_dic = sort_freq(distribution_dic)

    maximums = np.max(candidate_predictions, axis=1)

    if norm_mode == NORM_MODE_MAX:
        maximums = norm_list_max(maximums)
    else:
        maximums = norm_list_min_max(maximums)

    org_maximums = maximums.copy().tolist()
    for i in range(candidate_predictions.shape[0]):
        belongsto = np.argmax(candidate_predictions[i])
        if factor_mode == FACTOR_MODE_FREQ:
            maximums[i] = maximums[i] * 1.0 / rank_dic[belongsto]

    if debug:
        print ('active -before -after')
        for i in range(len(list(maximums))):
            print (org_maximums[i], list(maximums)[i])
    best = np.argsort(maximums)[:step]  # ascending order
    return best

def select_center(candidate_predictions, candidate_vectors,
                  train_labels, train_vectors,
                  distribution_dic,
                  step=100, factor_mode=FACTOR_MODE_BASE, norm_mode=NORM_MODE_MAX,
                  debug=False):
    rank_dic = sort_freq(distribution_dic)

    dist = pairwise_distances(candidate_vectors, train_vectors, metric="cosine")
    score = np.min(dist, axis=1)
    if norm_mode == NORM_MODE_MAX:
        score = norm_list_max(score)
    else:
        score = norm_list_min_max(score)

    for i in range(score.shape[0]):
        belongsto = np.argmax(candidate_predictions[i])
        if factor_mode == FACTOR_MODE_FREQ:
            score[i] /= rank_dic[belongsto]

    best = np.argsort(score)[::-1][:step]
    return best

def select_radius_multi_label_unlabel_centroid(
                                               candidate_predictions, candidate_vectors,
                                               train_labels, train_vectors,
                                               distribution_dic,
                              step=100, factor_mode=FACTOR_MODE_BASE, norm_mode=NORM_MODE_MAX,
                              debug=False, use_sigmoid=1):
    rank_dic = sort_freq(distribution_dic)

    entropies = []
    for prediction in candidate_predictions:
        entropy = 0
        for p in prediction:
            if p:
                entropy -= p * np.log2(p)
        entropies.append(entropy)
    entropies = np.array(entropies)

    # calculate the centroid of each class
    centroid = {}
    for index in range(train_vectors.shape[0]):
        belongsto = train_labels[index]
        if belongsto not in centroid.keys():
            centroid[belongsto] = [train_vectors[index]]
        else:
            centroid[belongsto].append(train_vectors[index])
    for index in range(candidate_vectors.shape[0]):
        belongsto = candidate_predictions[index].argmax()
        if belongsto not in centroid.keys():
            centroid[belongsto] = [candidate_vectors[index]]
        else:
            centroid[belongsto].append(candidate_vectors[index])

    for belongsto in centroid.keys():
        centroid[belongsto] = np.array(centroid[belongsto])
        center = np.mean(centroid[belongsto], axis=0)
        distances = euclidean_distances(np.array([center]), centroid[belongsto])

        distances = list(distances[0])
        distances.sort()
        if len(distances) <= 2:
            centroid[belongsto] = distances[-1]
        else:
            if distances[-1] - distances[-2] > distances[-2] - distances[-3]:
                centroid[belongsto] = distances[-2]
            else:
                centroid[belongsto] = distances[-1]

        if debug:
            print ('distances', distances)
            if len(distances) >= 2:
                print ('top1', distances[-1], 'top2', distances[-2])

    distances = {}
    for key in centroid.keys():
        if use_sigmoid:
            distances[key] = sigmoid(centroid[key])
        else:
            distances[key] = centroid[key] if centroid[key] != 0 else float('inf')

    # if debug:
    #     print("centroid information --class --original --sigmoid")
    #     keys = list(centroid.keys())
    #     inverse_keys = encoder.inverse_transform(keys)
    #     for i,key in enumerate(inverse_keys):
    #         print("(%s, %.3f, %.3f)" %(inverse_keys[i], centroid[keys[i]], distances[keys[i]]))

    estimates = []
    for i, vector in enumerate(candidate_predictions):
        belongsto = vector.argmax()
        estimates.append(entropies[i] * distances[belongsto])

    if norm_mode == NORM_MODE_MAX:
        estimates = norm_list_max(estimates)
    else:
        estimates = norm_list_min_max(estimates)

    for i, vector in enumerate(candidate_predictions):
        belongsto = vector.argmax()
        if factor_mode == FACTOR_MODE_FREQ:
            estimates[i] /= rank_dic[belongsto]

    estimates = np.array(estimates)
    best = np.argsort(estimates)[::-1][:step]
    return best
