from mmaction.datasets.pipelines import Compose
import torch.utils.data
import pandas as pd
import soundfile as sf
from scipy import signal
import numpy as np

class EPICDOMAIN(torch.utils.data.Dataset):
    def __init__(self, split='train', source=True, domain=['D1'], modality='rgb', cfg=None, cfg_flow=None, sample_dur=10, use_video=True, use_flow=True, use_audio=True, datapath='/path/to/EPIC-KITCHENS/'):
        self.base_path = datapath
        self.split = split
        self.modality = modality
        self.interval = 9
        self.sample_dur = sample_dur
        self.use_video = use_video
        self.use_audio = use_audio
        self.use_flow = use_flow
        self.domain_to_idx = {name: idx for idx, name in enumerate(domain)}

        # build the data pipeline
        if split == 'train':
            if self.use_video:
                train_pipeline = cfg.data.train.pipeline
                self.pipeline = Compose(train_pipeline)
            if self.use_flow:
                train_pipeline_flow = cfg_flow.data.train.pipeline
                self.pipeline_flow = Compose(train_pipeline_flow)
        else:
            if self.use_video:
                val_pipeline = cfg.data.val.pipeline
                self.pipeline = Compose(val_pipeline)
            if self.use_flow:
                val_pipeline_flow = cfg_flow.data.val.pipeline
                self.pipeline_flow = Compose(val_pipeline_flow)

        data1 = []
        split_dir = []
        for dom in domain:
            train_file = pd.read_pickle(self.base_path + 'MM-SADA_Domain_Adaptation_Splits/'+dom+"_"+split+".pkl")

            for _, line in train_file.iterrows():
                image = [dom + '/' + line['video_id'], line['start_frame'], line['stop_frame'], line['start_timestamp'],
                        line['stop_timestamp']]
                labels = line['verb_class']
                data1.append((image[0], image[1], image[2], image[3], image[4], int(labels), self.domain_to_idx[dom]))
                split_dir.append(split)

            if split == 'test' and not source:
                train_file = pd.read_pickle(self.base_path + 'MM-SADA_Domain_Adaptation_Splits/'+dom+"_train.pkl")
                for _, line in train_file.iterrows():
                    image = [dom + '/' + line['video_id'], line['start_frame'], line['stop_frame'], line['start_timestamp'],
                            line['stop_timestamp']]
                    labels = line['verb_class']
                    data1.append((image[0], image[1], image[2], image[3], image[4], int(labels), self.domain_to_idx[dom]))
                    split_dir.append('train')

        self.samples = data1
        self.split_dir = split_dir
        self.cfg = cfg
        self.cfg_flow = cfg_flow

    def __getitem__(self, index):
        label1 = self.samples[index][-2]
        domain_label = self.samples[index][-1]
        video_path = self.base_path +'rgb/'+ self.split_dir[index] + '/'+self.samples[index][0]
        flow_path = self.base_path +'flow/'+ self.split_dir[index] + '/'+self.samples[index][0]

        if self.use_video:
            filename_tmpl = self.cfg.data.train.get('filename_tmpl', 'frame_{:010}.jpg')
            modality = self.cfg.data.train.get('modality', 'RGB')
            start_index = self.cfg.data.train.get('start_index', int(self.samples[index][1]))
            data = dict(
                frame_dir=video_path,
                total_frames=int(self.samples[index][2] - self.samples[index][1]),
                # assuming files in ``video_path`` are all named with ``filename_tmpl``  # noqa: E501
                label=-1,
                start_index=start_index,
                filename_tmpl=filename_tmpl,
                modality=modality)
            data = self.pipeline(data)

        if self.use_flow:
            filename_tmpl_flow = self.cfg_flow.data.train.get('filename_tmpl', 'frame_{:010}.jpg')
            modality_flow = self.cfg_flow.data.train.get('modality', 'Flow')
            start_index_flow = self.cfg_flow.data.train.get('start_index', int(np.ceil(self.samples[index][1] / 2)))
            flow = dict(
                frame_dir=flow_path,
                total_frames=int((self.samples[index][2] - self.samples[index][1])/2),
                # assuming files in ``video_path`` are all named with ``filename_tmpl``  # noqa: E501
                label=-1,
                start_index=start_index_flow,
                filename_tmpl=filename_tmpl_flow,
                modality=modality_flow)
            flow = self.pipeline_flow(flow)

        if self.use_audio:
            audio_path = self.base_path + 'audio/' + self.split_dir[index] + '/' + self.samples[index][0] + '.wav'
            samples, samplerate = sf.read(audio_path)

            duration = len(samples) / samplerate

            fr_sec = self.samples[index][3].split(':')
            hour1 = float(fr_sec[0])
            minu1 = float(fr_sec[1])
            sec1 = float(fr_sec[2])
            fr_sec = (hour1 * 60 + minu1) * 60 + sec1

            stop_sec = self.samples[index][4].split(':')
            hour1 = float(stop_sec[0])
            minu1 = float(stop_sec[1])
            sec1 = float(stop_sec[2])
            stop_sec = (hour1 * 60 + minu1) * 60 + sec1

            start1 = fr_sec / duration * len(samples)
            end1 = stop_sec / duration * len(samples)
            start1 = int(np.round(start1))
            end1 = int(np.round(end1))
            samples = samples[start1:end1]

            resamples = samples[:160000]
            while len(resamples) < 160000:
                resamples = np.tile(resamples, 10)[:160000]

            resamples[resamples > 1.] = 1.
            resamples[resamples < -1.] = -1.
            frequencies, times, spectrogram = signal.spectrogram(resamples, samplerate, nperseg=512, noverlap=353)
            spectrogram = np.log(spectrogram + 1e-7)

            mean = np.mean(spectrogram)
            std = np.std(spectrogram)
            spectrogram = np.divide(spectrogram - mean, std + 1e-9)
            if self.split == 'train':
                noise = np.random.uniform(-0.05, 0.05, spectrogram.shape)
                spectrogram = spectrogram + noise
                start1 = np.random.choice(256 - self.interval, (1,))[0]
                spectrogram[start1:(start1 + self.interval), :] = 0

        video = data if self.use_video else 0
        flow = flow if self.use_flow else 0
        audio = spectrogram.astype(np.float32) if self.use_audio else 0

        return video, flow, audio, label1, domain_label

    def __len__(self):
        return len(self.samples)
