'''
DataLoader for training
'''

import glob, numpy, os, random, soundfile, torch
import sys

import tqdm
from scipy import signal
from torch.utils.data import Dataset, DataLoader


class ListDataset(Dataset):
    def __init__(self, lines, dictkeys, train_path):
        self.lines = lines
        self.dictkeys = dictkeys
        self.train_path = train_path

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        speaker_label = self.dictkeys[line.split()[0]]
        file_name = os.path.join(self.train_path, line.split()[1])
        return speaker_label, file_name


class ListDataset2(Dataset):
    def __init__(self, lines, dictkeys, train_path):
        self.lines = lines
        self.dictkeys = dictkeys
        self.train_path = train_path

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        id_speaker, path = line.split()
        idfile, _ = id_speaker.split("_")
        idfile = int(idfile)
        speaker_label = self.dictkeys[id_speaker]
        base_path = self.train_path[idfile]
        file_name = os.path.join(base_path, path)
        return speaker_label, file_name


class train_loader(Dataset):
    def __init__(self, train_list, train_path, musan_path, rir_path, num_frames, n_cpu, **kwargs):
        self.train_path = train_path
        self.num_frames = num_frames
        # Load and configure augmentation files
        self.noisetypes = ['noise', 'speech', 'music']
        self.noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
        self.numnoise = {'noise': [1, 1], 'speech': [3, 8], 'music': [1, 1]}
        self.noiselist = {}
        augment_files = glob.glob(os.path.join(musan_path, '*/*/*/*.wav'))
        for file in augment_files:
            if file.split('/')[-4] not in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)
        self.rir_files = glob.glob(os.path.join(rir_path, '*/*/*.wav'))
        # Load data & labels
        self.data_list = []
        self.data_label = []
        lines = open(train_list).read().splitlines()
        dictkeys = list(set([x.split()[0] for x in lines]))
        dictkeys.sort()
        dictkeys = {key: ii for ii, key in enumerate(dictkeys)}
        print(f"BEGIN Initializing data loader: ", flush=True)
        sys.stdout.flush()
        dataset = ListDataset(lines, dictkeys, train_path)
        loader = DataLoader(dataset, num_workers=n_cpu, batch_size=128)
        for index, batch in tqdm.tqdm(enumerate(loader, start=1), total=len(loader)):
            speaker_labels, file_names = batch
            for i in range(len(speaker_labels)):
                speaker_label = speaker_labels[i]
                file_name = file_names[i]
                if speaker_label is not None:
                    self.data_label.append(speaker_label)
                    self.data_list.append(file_name)
            sys.stdout.flush()
        print(f"END Initializing data loader: ", flush=True)

    def __getitem__(self, index):
        # Read the utterance and randomly select the segment
        audio, sr = soundfile.read(self.data_list[index])
        length = self.num_frames * 160 + 240
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), 'wrap')
        start_frame = numpy.int64(random.random() * (audio.shape[0] - length))
        audio = audio[start_frame:start_frame + length]
        audio = numpy.stack([audio], axis=0)
        # Data Augmentation
        augtype = random.randint(0, 5)
        if augtype == 0:  # Original
            audio = audio
        elif augtype == 1:  # Reverberation
            audio = self.add_rev(audio)
        elif augtype == 2:  # Babble
            audio = self.add_noise(audio, 'speech')
        elif augtype == 3:  # Music
            audio = self.add_noise(audio, 'music')
        elif augtype == 4:  # Noise
            audio = self.add_noise(audio, 'noise')
        elif augtype == 5:  # Television noise
            audio = self.add_noise(audio, 'speech')
            audio = self.add_noise(audio, 'music')
        return torch.FloatTensor(audio[0]), self.data_label[index]

    def __len__(self):
        return len(self.data_list)

    def add_rev(self, audio):
        rir_file = random.choice(self.rir_files)
        rir, sr = soundfile.read(rir_file)
        rir = numpy.expand_dims(rir.astype(numpy.float64), 0)
        rir = rir / numpy.sqrt(numpy.sum(rir ** 2))
        return signal.convolve(audio, rir, mode='full')[:, :self.num_frames * 160 + 240]

    def add_noise(self, audio, noisecat):
        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2) + 1e-4)
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio, sr = soundfile.read(noise)
            length = self.num_frames * 160 + 240
            if noiseaudio.shape[0] <= length:
                shortage = length - noiseaudio.shape[0]
                noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
            start_frame = numpy.int64(random.random() * (noiseaudio.shape[0] - length))
            noiseaudio = noiseaudio[start_frame:start_frame + length]
            noiseaudio = numpy.stack([noiseaudio], axis=0)
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2) + 1e-4)
            noisesnr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = numpy.sum(numpy.concatenate(noises, axis=0), axis=0, keepdims=True)
        return noise + audio


class TrainDatasetMulti(Dataset):
    def __init__(self, train_list, train_path, musan_path, rir_path, num_frames, n_cpu, **kwargs):
        self.train_path = train_path
        self.num_frames = num_frames
        # Load and configure augmentation files
        self.noisetypes = ['noise', 'speech', 'music']
        self.noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
        self.numnoise = {'noise': [1, 1], 'speech': [3, 8], 'music': [1, 1]}
        self.noiselist = {}
        augment_files = glob.glob(os.path.join(musan_path, '*/*/*/*.wav'))
        for file in augment_files:
            if file.split('/')[-4] not in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)
        self.rir_files = glob.glob(os.path.join(rir_path, '*/*/*.wav'))
        # Load data & labels
        train_list = train_list.strip().split('\n')
        train_list = [tl.strip() for tl in train_list if len(tl.strip()) > 0]
        train_path = train_path.strip().split('\n')
        train_path = [tp.strip() for tp in train_path if len(tp.strip()) > 0]

        self.data_list = {}
        self.data_label = {}

        for idx_file, train_list_ in enumerate(train_list):
            self.data_list[idx_file] = []
            self.data_label[idx_file] = []
            train_list_ = train_list_.strip()
            lines = open(train_list_).read().splitlines()
            dictkeys = list(set([x.split()[0] for x in lines]))
            dictkeys.sort()
            dictkeys = {key: ii for ii, key in enumerate(dictkeys)}
            print(f"BEGIN Initializing data loader: {train_list_}", flush=True)
            train_path_ = train_path[idx_file].strip()
            dataset = ListDataset(lines, dictkeys, train_path_)
            loader = DataLoader(dataset, num_workers=n_cpu, batch_size=128)
            for index, batch in tqdm.tqdm(enumerate(loader, start=1), total=len(loader)):
                speaker_labels, file_names = batch
                for i in range(len(speaker_labels)):
                    speaker_label = speaker_labels[i]
                    file_name = file_names[i]
                    if speaker_label is not None:
                        self.data_label[idx_file].append(speaker_label)
                        self.data_list[idx_file].append(file_name)
            print(f"END Initializing data loader: {train_list_}", flush=True)

        max_length = max(len(lst) for lst in self.data_list.values())
        for key in self.data_list:
            value = self.data_list[key].copy()
            value_ = value * (max_length // len(value)) + value[:(max_length % len(value))]
            self.data_list[key] = value_
            label = self.data_label[key].copy()
            label_ = label * (max_length // len(label)) + label[:(max_length % len(label))]
            self.data_label[key] = label_

    def __getitem__(self, index):
        # Read the utterance and randomly select the segment
        result = []
        for i, data_list_ in self.data_list.items():
            data_label_ = self.data_label[i]
            audio, sr = soundfile.read(data_list_[index])
            length = self.num_frames * 160 + 240
            if audio.shape[0] <= length:
                shortage = length - audio.shape[0]
                audio = numpy.pad(audio, (0, shortage), 'wrap')
            start_frame = numpy.int64(random.random() * (audio.shape[0] - length))
            audio = audio[start_frame:start_frame + length]
            audio = numpy.stack([audio], axis=0)
            # Data Augmentation
            augtype = random.randint(0, 5)
            if augtype == 0:  # Original
                audio = audio
            elif augtype == 1:  # Reverberation
                audio = self.add_rev(audio)
            elif augtype == 2:  # Babble
                audio = self.add_noise(audio, 'speech')
            elif augtype == 3:  # Music
                audio = self.add_noise(audio, 'music')
            elif augtype == 4:  # Noise
                audio = self.add_noise(audio, 'noise')
            elif augtype == 5:  # Television noise
                audio = self.add_noise(audio, 'speech')
                audio = self.add_noise(audio, 'music')

            result.append(torch.FloatTensor(audio[0]))
            result.append(data_label_[index])

        return tuple(result)

    def __len__(self):
        return len(self.data_list[0])

    def add_rev(self, audio):
        rir_file = random.choice(self.rir_files)
        rir, sr = soundfile.read(rir_file)
        rir = numpy.expand_dims(rir.astype(numpy.float64), 0)
        rir = rir / numpy.sqrt(numpy.sum(rir ** 2))
        return signal.convolve(audio, rir, mode='full')[:, :self.num_frames * 160 + 240]

    def add_noise(self, audio, noisecat):
        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2) + 1e-4)
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio, sr = soundfile.read(noise)
            length = self.num_frames * 160 + 240
            if noiseaudio.shape[0] <= length:
                shortage = length - noiseaudio.shape[0]
                noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
            start_frame = numpy.int64(random.random() * (noiseaudio.shape[0] - length))
            noiseaudio = noiseaudio[start_frame:start_frame + length]
            noiseaudio = numpy.stack([noiseaudio], axis=0)
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2) + 1e-4)
            noisesnr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = numpy.sum(numpy.concatenate(noises, axis=0), axis=0, keepdims=True)
        return noise + audio


def append_suffix_to_list(lines, suffix):
    result = []
    for line in lines:
        id_speaker, path = line.split()
        id_speaker = f"{suffix}_{id_speaker.strip()}"
        path = path.strip()
        result.append(f"{id_speaker} {path}")
    return result


class TrainDatasetMulti2(Dataset):
    def __init__(self, train_list, train_path, musan_path, rir_path, num_frames, n_cpu, **kwargs):
        self.train_path = train_path
        self.num_frames = num_frames
        # Load and configure augmentation files
        self.noisetypes = ['noise', 'speech', 'music']
        self.noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
        self.numnoise = {'noise': [1, 1], 'speech': [3, 8], 'music': [1, 1]}
        self.noiselist = {}
        augment_files = glob.glob(os.path.join(musan_path, '*/*/*/*.wav'))
        for file in augment_files:
            if file.split('/')[-4] not in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)
        self.rir_files = glob.glob(os.path.join(rir_path, '*/*/*.wav'))
        # Load data & labels
        train_list = train_list.strip().split('\n')
        train_list = [tl.strip() for tl in train_list if len(tl.strip()) > 0]
        train_path = train_path.strip().split('\n')
        train_path = [tp.strip() for tp in train_path if len(tp.strip()) > 0]

        self.data_list = []
        self.data_label = []

        lines = []
        for idx_file, train_list_ in enumerate(train_list):
            train_list_ = train_list_.strip()
            lines_ = open(train_list_).read().splitlines()
            lines.extend(append_suffix_to_list(lines_, idx_file))

        dictkeys = list(set([x.split()[0].strip() for x in lines]))
        dictkeys.sort()
        dictkeys = {key: ii for ii, key in enumerate(dictkeys)}
        dataset = ListDataset2(lines, dictkeys, train_path)
        loader = DataLoader(dataset, num_workers=n_cpu, batch_size=128)
        print(f"BEGIN Initializing data loader: ", flush=True)
        for index, batch in tqdm.tqdm(enumerate(loader, start=1), total=len(loader)):
            speaker_labels, file_names = batch
            for i in range(len(speaker_labels)):
                speaker_label = speaker_labels[i]
                file_name = file_names[i]
                if speaker_label is not None:
                    self.data_label.append(speaker_label)
                    self.data_list.append(file_name)

        print(f"END Initializing data loader: ", flush=True)

    def __getitem__(self, index):
        # Read the utterance and randomly select the segment
        audio, sr = soundfile.read(self.data_list[index])
        length = self.num_frames * 160 + 240
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), 'wrap')
        start_frame = numpy.int64(random.random() * (audio.shape[0] - length))
        audio = audio[start_frame:start_frame + length]
        audio = numpy.stack([audio], axis=0)
        # Data Augmentation
        augtype = random.randint(0, 5)
        if augtype == 0:  # Original
            audio = audio
        elif augtype == 1:  # Reverberation
            audio = self.add_rev(audio)
        elif augtype == 2:  # Babble
            audio = self.add_noise(audio, 'speech')
        elif augtype == 3:  # Music
            audio = self.add_noise(audio, 'music')
        elif augtype == 4:  # Noise
            audio = self.add_noise(audio, 'noise')
        elif augtype == 5:  # Television noise
            audio = self.add_noise(audio, 'speech')
            audio = self.add_noise(audio, 'music')
        return torch.FloatTensor(audio[0]), self.data_label[index]

    def __len__(self):
        return len(self.data_list)

    def add_rev(self, audio):
        rir_file = random.choice(self.rir_files)
        rir, sr = soundfile.read(rir_file)
        rir = numpy.expand_dims(rir.astype(numpy.float64), 0)
        rir = rir / numpy.sqrt(numpy.sum(rir ** 2))
        return signal.convolve(audio, rir, mode='full')[:, :self.num_frames * 160 + 240]

    def add_noise(self, audio, noisecat):
        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2) + 1e-4)
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio, sr = soundfile.read(noise)
            length = self.num_frames * 160 + 240
            if noiseaudio.shape[0] <= length:
                shortage = length - noiseaudio.shape[0]
                noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
            start_frame = numpy.int64(random.random() * (noiseaudio.shape[0] - length))
            noiseaudio = noiseaudio[start_frame:start_frame + length]
            noiseaudio = numpy.stack([noiseaudio], axis=0)
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2) + 1e-4)
            noisesnr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = numpy.sum(numpy.concatenate(noises, axis=0), axis=0, keepdims=True)
        return noise + audio
