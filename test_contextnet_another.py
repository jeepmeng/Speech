import unittest

from contextnet.model import ContextNet
import torch
import dataset
import os
import wave
import yaml
from speech_features import speech_features
from dataset import *
import torch.optim as optim

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








# batch_size = 1
# input_size = 80
# seq_length = 500
#
cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
#
model = ContextNet(
            model_size='medium',
            num_vocabs=1426,
        ).to(device)

X = np.zeros((1, 500, 80), dtype=np.float)
y = np.zeros((1, 1426), dtype=np.int16)


# print(X[0, 0:266].shape)


cl = DataLoader()
wav_signal, sample_rate, data_label = cl.get_data(5)
ll = speech_features.MFCC()
data_input = ll.run(wav_signal, sample_rate)
print(data_label)
# data_input = data_input.reshape(data_input.shape[0], data_input.shape[1], 1)
# print(data_input.shape)
# print(data_label.shape)
# print(len(data_input))

X[0, 0:data_input.shape[0], 0:data_input.shape[1]] = data_input
y[0, 0:len(data_label)] = data_label
inputs = torch.FloatTensor(X).to(device)
# inputs = inputs.unsqueeze(0)
input_lengths = torch.IntTensor([inputs.shape[1]]).to(device)


targets = torch.LongTensor(y).to(device)
target_lengths = torch.LongTensor([data_label.shape[0]]).to(device)
#
print('inputs.shape--------{}'.format(inputs.shape))
print('input_lengths.shape--------{}'.format(input_lengths.shape))
print('targets.shape--------{}'.format(targets.shape))
print('target_lengths.shape--------{}'.format(target_lengths.shape))



# batch_size = 1
# labels = np.zeros((batch_size, 1), dtype=np.float)
# data_count = cl.get_data_count()
# index = 0
#
#
# X = np.zeros((batch_size,) + self.speech_model.input_shape, dtype=np.float)
# y = np.zeros((batch_size, self.max_label_length), dtype=np.int16)
# input_length = []
# label_length = []
#
#\



outputs = model(inputs, input_lengths, targets, target_lengths)

# outputs = model.recognize(inputs, input_lengths)
print(outputs.shape)


# if __name__ == '__main__':
#     # unittest.main()
#     cl = DataLoader()
#     # cl._load_data()
#
#     wav_signal, sample_rate, data_label = cl.get_data(5)
#     # print('wav_signal-----{}'.format(len(wav_signal[0])))
#     # print('sample_rate-----{}'.format(sample_rate))
#     # print('data_label-----{}'.format(len(data_label)))
#
#
#     ll = speech_features.MFCC()
#     data_input = ll.run(wav_signal)
#
#     print('data_input-----{}'.format(len(data_input)))
#     # config = load_config_file(pth)
#     # print(config['dic_filename'])
#
#     print('done')