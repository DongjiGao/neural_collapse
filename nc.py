#!/usr/bin/env python3

# 2021 Dongji Gao

import argparse
import soundfile as sf
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from datasets import load_dataset
from torch.linalg import lstsq
from transformers import Wav2Vec2Model, Wav2Vec2ForCTC, Wav2Vec2Processor


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        type=str,
        default="dgao/librispeech_nc_test",
        help="audio data")
    parser.add_argument(
        '--model',
        type=str,
        default="facebook/wav2vec2-large-960h",
        help="pre-trained model")
    parser.add_argument(
        'output',
        type=str,
        help="output file")
    return parser


def plot_angle(feature_means, weights, ids_sorted):
    cos = torch.nn.CosineSimilarity()
    feature_angles = list()
    weight_angles = list()

    num_feature_means = feature_means.shape[0]
    print(num_feature_means)
    assert num_feature_means == len(ids_sorted)

    for i in range(num_feature_means - 1):
        for j in range(i + 1, num_feature_means):
            feature_angle = cos(feature_means[i].unsqueeze(dim=0),
                                feature_means[j].unsqueeze(dim=0))
            feature_angles.append(feature_angle.item())
            weight_angle = cos(weights[ids_sorted[i]].unsqueeze(dim=0),
                               weights[ids_sorted[j]].unsqueeze(dim=0))
            weight_angles.append(weight_angle.item())

    # plot
    feature_angles = np.array(feature_angles)
    weight_angles = np.array(weight_angles)

    c = feature_angles ** 2 + weight_angles ** 2
    fig, ax = plt.subplots()
    ax.scatter(feature_angles, weight_angles, s=5, c=c, cmap=plt.cm.coolwarm, zorder=10)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    fig.savefig("librispeech.png")


def analysis(features, class_boundary):
    features = torch.stack(features)
    global_feature_mean = features.mean(dim=0)

    num_features, feature_size = features.shape
    num_classes = len(class_boundary)
    last_boundary = class_boundary[-1][-1]

    assert num_features == last_boundary

    intra_feature_means = list()
    for _, start, end in class_boundary:
        intra_feature_mean = features[start:end].mean(dim=0)
        intra_feature_means.append(intra_feature_mean)
        # here we substract the mean inplace
        features[start:end] -= intra_feature_mean

    intra_feature_means = torch.stack(intra_feature_means)

    intra_covariance_matrix = torch.zeros(feature_size, feature_size)
    for i in range(num_features):
        cur_vector = features[i].unsqueeze(1)
        cur_covariance = torch.matmul(cur_vector, cur_vector.T)
        intra_covariance_matrix += cur_covariance
    intra_covariance_matrix /= num_features

    inter_covariance_matrix = torch.zeros(feature_size, feature_size)
    for i in range(num_classes):
        # unsqueeze for torch.matmul()
        cur_vector_mean = (intra_feature_means[i] - global_feature_mean).unsqueeze(1)
        cur_covariance = torch.matmul(cur_vector_mean, cur_vector_mean.T)
        inter_covariance_matrix += cur_covariance
    inter_covariance_matrix /= num_classes

    result = torch.trace(lstsq(inter_covariance_matrix,
                               intra_covariance_matrix).solution) / num_classes
    print(result)
    return intra_feature_means


def main():
    args = get_parser().parse_args()
    device = torch.device("cpu")

    data = args.data
    pretrained_model = args.model
    output = args.output

    # processor raw wav and text files
    processor = Wav2Vec2Processor.from_pretrained(pretrained_model,
                                                  feature_size=3)

    model = Wav2Vec2Model.from_pretrained(pretrained_model)
    ctc_model = Wav2Vec2ForCTC.from_pretrained(pretrained_model)

    def map_to_array(batch):
        speech, _ = sf.read(batch["file"])
        batch["speech"] = speech
        return batch

    def map_to_result(batch):
        model.to(device)
        ctc_model.to(device)

        input_values = processor(batch["speech"],
                                 sampling_rate=16000,
                                 return_tensors="pt").input_values.to(device)
        with torch.no_grad():
            logits = ctc_model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1).squeeze()
        # get feature
        model_output = model(input_values, output_hidden_states=True)
        feature = model_output.extract_features.squeeze()
        last_hidden_state = model_output.last_hidden_state.squeeze()

        predicted_ids_list.append(predicted_ids)
        last_hidden_states_list.append(last_hidden_state)

    # decode
    predicted_ids_list = list()
    last_hidden_states_list = list()

    librispeech_ds = load_dataset(data, "clean", split="validation")
    librispeech_ds = librispeech_ds.map(map_to_array)
    result = librispeech_ds.map(map_to_result)


    predicted_ids = torch.cat(predicted_ids_list)
    last_hidden_states = torch.cat(last_hidden_states_list)

    num_features, feature_size = last_hidden_states.shape
    num_ids = predicted_ids.shape[0]
    assert num_features == num_ids

    context_features = list()
    context_features_dict = defaultdict(list)
    class_boundary = list()

    start = 0
    ids_sorted = list()
    for i in range(num_features):
        id = predicted_ids[i].item()
        if id != 0:
            context_features_dict[id].append(last_hidden_states[i])
            if id not in ids_sorted:
                ids_sorted.append(id)
    ids_sorted.sort()

    for id in ids_sorted:
        context_feature = context_features_dict[id]
        context_features += context_feature
        class_boundary.append((id, start, start + len(context_feature)))
        start += len(context_feature)

        # simple test
        #    a_0 = torch.tensor([0.9, 1.0, 1.1])
        #    a_1 = torch.tensor([0.8, 1.1, 0.9])
        #    b_0 = torch.tensor([11, 10, 9])
        #    b_1 = torch.tensor([12, 11, 12])
        #    b_2 = torch.tensor([13, 10, 9])
        #    c_0 = torch.tensor([100, 110, 95])
        #    c_1 = torch.tensor([98, 99, 102])
        #    context_features = [a_0, a_1, b_0, b_1, b_2, c_0, c_1]
        #    class_boundary = [(0,2), (2,5), (5,7)]

    intra_feature_means = analysis(context_features, class_boundary)

    num_intra_feature_means = intra_feature_means.shape[0]
    num_classes = len(class_boundary)
    assert num_intra_feature_means == num_classes

    for w in ctc_model.lm_head.parameters():
        weights = w
        break

    plot_angle(intra_feature_means, weights, ids_sorted)


if __name__ == "__main__":
    main()
