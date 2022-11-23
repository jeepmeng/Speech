import torch
import numpy as np
import os
import soundfile
import glob


pinyin_dict = r'dict.txt'
train_data_dict = r'/Users/liufucong/Downloads/ltxm/ContextNet-master/datalist/st-cmds/train.wav.txt'
dev_data_dict = r'/Users/liufucong/Downloads/ltxm/ContextNet-master/datalist/st-cmds/dev.wav.txt'
test_data_dict = r'/Users/liufucong/Downloads/ltxm/ContextNet-master/datalist/st-cmds/test.wav.txt'
print(train_data_dict,'\n',dev_data_dict,'\n',test_data_dict)











def _get_output_sequence(vocab, transcript):
    if isinstance(transcript, bytes):
        transcript = transcript.decode('utf-8')
    labels = [vocab[char] if char in vocab else vocab['<unk>'] for char in transcript]
    return np.array(labels, dtype=np.int32), np.array(len(labels), dtype=np.int32)

def create_dataset(librispeech_dir, data_key, vocab, mean=None, std_dev=None, num_feats=40):
    """ librispeech_dir (str): path to directory containing librispeech data
        data_key (str) : train / dev / test
        mean (str|None) : path to file containing mean of librispeech training data
        std_dev (str|None) : path to file containing std_dev of librispeech training data
        num_feats (int) : input feature dimension

        Returns : tf.data.dataset instance
    """
    vocab = eval(open(vocab).read().strip())
    if mean:
        mean = np.loadtxt(mean).astype("float32")
    if std_dev:
        std_dev = np.loadtxt(std_dev).astype("float32")

    def _generate_librispeech_examples():
        """Generate examples from a Librispeech directory."""
        audios, transcripts = [], []
        transcripts_glob = os.path.join(librispeech_dir, "%s*/*/*/*.txt" % data_key)
        for transcript_file in glob.glob(transcripts_glob):
            path = os.path.dirname(transcript_file)
            for line in open(transcript_file).read().strip().splitlines():
                line = line.strip()
                key, transcript = line.split(" ", 1)
                audio_file = os.path.join(path, "%s.flac" % key)
                audios.append(audio_file)
                transcripts.append(transcript)
        return audios, transcripts

    def _extract_audio_features(audio_file):
        audio, sample_rate = soundfile.read(audio_file)
        feats = _get_audio_features_mfcc(audio, sample_rate)

        if mean is not None:
            feats = feats - mean
        if std_dev is not None:
            feats = feats / std_dev
        return feats, np.array(feats.shape[0], dtype=np.int32)

    def _extract_output_sequence(transcript):
        return _get_output_sequence(vocab, transcript)

    def _prepare(audio_file, transcript):
        audio_feats, timesteps = _extract_audio_features(audio_file)
        output_seq, seq_len = _extract_output_sequence(transcript)
        return audio_feats, output_seq, timesteps, seq_len

    audios, transcripts = _generate_librispeech_examples()
    dataset = tf.data.Dataset.from_tensor_slices((audios, transcripts))
    dataset = dataset.shuffle(1000000)

    dataset = dataset.map(lambda audio_file, transcript: \
                          tf.numpy_function(_prepare, [audio_file, transcript],
                          (tf.float32, tf.int32, tf.int32, tf.int32)),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Remove utterances which have >300 chars
    dataset = dataset.filter(lambda x, y, _, y_len: y_len <= 300)
    dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(
                  element_length_func=lambda x, y, x_len, _: x_len,
                  bucket_boundaries=[500, 1000, 1250, 1500, 2000],
                  bucket_batch_sizes=[32, 16, 16, 8, 8, 4],
                  padded_shapes=([None, num_feats], [None], [], [])))
    return dataset


class DataLoader:
    def __init__(self, dataset_type = 'train'):
        self.dataset_type = dataset_type

        self.data_list = list()
        self.wav_dict = dict()
        self.label_dict = dict()
        self.pinyin_list = list()#拼音索引
        self.pinyin_dict = dict()#汉字段长度
        self._load_data()

    def _load_data(self):
        config = load_config_file(DEFAULT_CONFIG_FILENAME)

        self.pinyin_list, self.pinyin_dict = load_pinyin_dict(config['dic_filename'])

        # for index in range(len(config['dataset'][self.dataset_type])):
        for index in DATA_SET_NAME:
            idx = index+'_'+self.dataset_type
            print(type(idx),idx)
            filename_datalist = config['dataset'][self.dataset_type][idx]['data_list']
            filename_datapath = config['dataset'][self.dataset_type][idx]['data_pth']
            with open(filename_datalist, 'r', encoding='utf-8') as file_pointer:
                lines = file_pointer.read().split('\n')
                for line in lines:
                    if len(line) == 0:
                        continue
                    tokens = line.split(' ')
                    self.data_list.append(tokens[0])
                    self.wav_dict[tokens[0]] = os.path.join(filename_datapath, tokens[1])

            filename_labellist = config['dataset'][self.dataset_type][idx]['label_list']
            with open(filename_labellist, 'r', encoding='utf-8') as file_pointer:
                lines = file_pointer.read().split('\n')
                for line in lines:
                    if len(line) == 0:
                        continue
                    tokens = line.split(' ')
                    self.label_dict[tokens[0]] = tokens[1:]


    def get_data_count(self) -> int:
        """
        获取数据集总数量
        """
        return len(self.data_list)


    def get_data(self, index: int) -> tuple:
        """
        按下标获取一条数据
        """
        mark = self.data_list[index]

        wav_signal, sample_rate, _, _ = read_wav_data(self.wav_dict[mark])
        labels = list()
        # print('label_dict[mark]',self.label_dict[mark])
        for item in self.label_dict[mark]:
            if len(item) == 0:
                continue
            labels.append(self.pinyin_dict[item])

        data_label = np.array(labels)
        return wav_signal, sample_rate, data_label



def _get_audio_features_mfcc(audio, sample_rate, window_len=0.025, step_len=0.010, num_feature_filters=40):
  """
  :param numpy.ndarray audio: raw audio samples, shape (audio_len,)
  :param int sample_rate: e.g. 22050
  :param float window_len: in seconds
  :param float step_len: in seconds
  :param int num_feature_filters:
  :return: (audio_len // int(step_len * sample_rate), num_feature_filters), float32
  :rtype: numpy.ndarray
  """
  import librosa
  mfccs = librosa.feature.mfcc(
    audio, sr=sample_rate,
    n_mfcc=num_feature_filters,
    hop_length=int(step_len * sample_rate), n_fft=int(window_len * sample_rate))
  librosa_version = librosa.__version__.split(".")
  if int(librosa_version[0]) >= 1 or (int(librosa_version[0]) == 0 and int(librosa_version[1]) >= 7):
    rms_func = librosa.feature.rms
  else:
    rms_func = librosa.feature.rmse
  energy = rms_func(
    audio,
    hop_length=int(step_len * sample_rate), frame_length=int(window_len * sample_rate))
  mfccs[0] = energy  # replace first MFCC with energy, per convention
  assert mfccs.shape[0] == num_feature_filters  # (dim, time)
  mfccs = mfccs.transpose().astype("float32")  # (time, dim)
  return mfccs


if __name__ == '__main__':

    a,b = load_pinyin_dict(pinyin_dict)
    print(a)
    print(b)
    print('done')

    a = DataLoader('train')
    # print(a.data_list)
    wav_signal, sample_rate, data_label = a.get_data(0)
    print('wav_signal', len(wav_signal[0]))
    print('sample_rate', sample_rate)
    print('data_label', len(data_label))