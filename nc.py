#!/usr/bin/env python3

# 2021 Dongji Gao

import argparse
import soundfile as sf
import torch
from collections import defaultdict
from datasets import load_dataset
from torch.linalg import lstsq
from transformers import Wav2Vec2Model, Wav2Vec2ForCTC, Wav2Vec2Processor


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        type=str,
        default="patrickvonplaten/librispeech_asr_dummy",
        help="audio data")
    parser.add_argument(
        '--model',
        type=str,
        default="facebook/wav2vec2-base-960h",
        help="pre-trained model")
    parser.add_argument(
        'output',
        type=str,
        help="output file")
    return parser


def analysis(features, class_boundary):
    features = torch.stack(features)
    global_feature_mean = features.mean(dim=0)

    num_features, feature_size = features.shape
    num_classes = len(class_boundary)
    last_boundary = class_boundary[-1][-1]

    assert num_features == last_boundary

    intra_feature_means = list()
    for start, end in class_boundary:
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
        cur_vector_mean = (intra_feature_mean[i] - global_feature_mean).unsqueeze(1)
        cur_covariance = torch.matmul(cur_vector_mean, cur_vector_mean.T)
        inter_covariance_matrix += cur_covariance
    inter_covariance_matrix /= num_classes

    result = torch.trace(lstsq(inter_covariance_matrix,
                               intra_covariance_matrix).solution) / num_classes
    print(result)


def main():
    args = get_parser().parse_args()

    data = args.data
    pretrained_model = args.model
    output = args.output

    # processor raw wav and text files
    processor = Wav2Vec2Processor.from_pretrained(pretrained_model,
                                                  feature_size=3)

    model = Wav2Vec2Model.from_pretrained(pretrained_model)
    ctc_model = Wav2Vec2ForCTC.from_pretrained(pretrained_model)

    librispeech_samples_ds = load_dataset(data, "clean", split="validation")
    audio_input, sample_rate = sf.read(librispeech_samples_ds[0]["file"])

    input_values = processor(audio_input, sampling_rate=sample_rate,
                             return_tensors="pt").input_values

    # get feature
    model_output = model(input_values, output_hidden_states=True)
    features = model_output.extract_features.squeeze()
    last_hidden_states = model_output.last_hidden_state.squeeze()

    # get label
    logits = ctc_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1).squeeze()

    num_features, feature_size = last_hidden_states.shape
    num_ids = predicted_ids.shape[0]
    assert num_features == num_ids

    context_features = list()
    context_features_dict = defaultdict(list)
    class_boundary = list()

    start = 0
    for i in range(num_features):
        id = predicted_ids[i].item()
        if id != 0:
            context_features_dict[id].append(last_hidden_states[i])

    for _, context_feature in context_features_dict.items():
        context_features += context_feature
        class_boundary.append((start, start + len(context_feature)))
        start += len(context_feature)

#    a_0 = torch.tensor([0.9, 1.0, 1.1])
#    a_1 = torch.tensor([0.8, 1.1, 0.9])
#    b_0 = torch.tensor([11, 10, 9])
#    b_1 = torch.tensor([12, 11, 12])
#    b_2 = torch.tensor([13, 10, 9])
#    c_0 = torch.tensor([100, 110, 95])
#    c_1 = torch.tensor([98, 99, 102])

    #    context_features = [a_0, a_1, b_0, b_1, b_2, c_0, c_1]
    #    class_boundary = [(0,2), (2,5), (5,7)]
    analysis(context_features, class_boundary)


if __name__ == "__main__":
    main()
