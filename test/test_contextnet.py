import unittest

from contextnet.model import ContextNet
import torch
import dataset
import os
import wave
import yaml
from speech_features import speech_features
from dataset import *
import torchsummary

# class TestContextNet(unittest.TestCase):
#     def test_forward(self):
#         batch_size = 3
#         seq_length = 500
#         input_size = 80
#
#         cuda = torch.cuda.is_available()
#         device = torch.device('cuda' if cuda else 'cpu')
#
#         model = ContextNet(
#             model_size='medium',
#             num_vocabs=10,
#         ).to(device)
#
#         inputs = torch.FloatTensor(batch_size, seq_length, input_size).to(device)
#         input_lengths = torch.IntTensor([500, 450, 350]).to(device)
#         targets = torch.LongTensor([[1, 3, 3, 3, 3, 3, 4, 5, 6, 2],
#                                     [1, 3, 3, 3, 3, 3, 4, 5, 2, 0],
#                                     [1, 3, 3, 3, 3, 3, 4, 2, 0, 0]]).to(device)
#         target_lengths = torch.LongTensor([9, 8, 7]).to(device)
#
#         outputs = model(inputs, input_lengths, targets, target_lengths)
#
#         print(outputs.size())  # torch.Size([3, 59, 9, 10])
#
#     def test_recognize(self):
#         batch_size = 3
#         seq_length = 500
#         input_size = 80
#
#         cuda = torch.cuda.is_available()
#         device = torch.device('cuda' if cuda else 'cpu')
#
#         model = ContextNet(
#             model_size='medium',
#             num_vocabs=10,
#         ).to(device)
#
#         inputs = torch.FloatTensor(batch_size, seq_length, input_size).to(device)
#         input_lengths = torch.IntTensor([500, 450, 350]).to(device)
#
#         outputs = model.recognize(inputs, input_lengths)
#
#         print(outputs.size())  # torch.Size([3, 59])


if __name__ == '__main__':
    # unittest.main()
    # cl = DataLoader()
    # # cl._load_data()
    #
    # wav_signal, sample_rate, data_label = cl.get_data(5)
    # # print('wav_signal-----{}'.format(len(wav_signal[0])))
    # # print('sample_rate-----{}'.format(sample_rate))
    # # print('data_label-----{}'.format(len(data_label)))
    #
    #
    # ll = speech_features.MFCC()
    # data_input = ll.run(wav_signal)
    #
    # print('data_input-----{}'.format(len(data_input)))
    # # config = load_config_file(pth)
    # # print(config['dic_filename'])
    #
    # print('done')

    batch_size = 3
    seq_length = 500
    input_size = 80

    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    model = ContextNet(
        model_size='medium',
        num_vocabs=10,
    ).to(device)

    inputs = torch.FloatTensor(batch_size, seq_length, input_size).to(device)
    input_lengths = torch.IntTensor([500, 450, 350]).to(device)
    targets = torch.LongTensor([[1, 3, 3, 3, 3, 3, 4, 5, 6, 2],
                                [1, 3, 3, 3, 3, 3, 4, 5, 2, 0],
                                [1, 3, 3, 3, 3, 3, 4, 2, 0, 0]]).to(device)
    target_lengths = torch.LongTensor([9, 8, 7]).to(device)

    outputs = model(inputs, input_lengths, targets, target_lengths)
    print(inputs.shape)
    print(input_lengths.shape)
    print(targets.shape)
    print(target_lengths.shape)
    print(outputs.shape)

    # torchsummary.summary(model(inputs, input_lengths, targets, target_lengths),(1,500,80))
    # print('parameters_count:',count_parameters(model))