import json
import os
import random

import numpy as np
import torch
import torchaudio
from torch.nn import functional as F


class AudioDataset(torch.utils.data.Dataset):

    def __init__(self,
                 dataset_dir,
                 dataset_list,
                 sample_rate=8000):
            """
            Initialize the Dataset object.

            Args:
                dataset_dir (str): The directory path of the dataset.
                dataset_list (list): The list of dataset files.
                sample_rate (int, optional): The sample rate of the audio files. Defaults to 8000.
            """
            self.dataset_dir = dataset_dir
            self.dataset_list = dataset_list
            self.sample_rate  = sample_rate
        
    def __getitem__(self, index):

        file_path = os.path.join(self.dataset_dir, self.dataset_list[index])
        audio, sr  = torchaudio.load(file_path)

        if sr != self.sample_rate:
            raise ValueError(f"Invalid sample rate: expected {self.sample_rate}, got {sr}")
        return audio
    
    def __len__(self):
        return len(self.dataset_list)
    
class LibriMixDataset(torch.utils.data.Dataset):


    def __init__(self,
                 dataset_dir,
                 mix_num=2,
                 segment=None,
                 sample_rate=8000,
                 mode:str="clean",
                 dynamic_mix=False):
        """
        Initialize the Dataset object.

        Args:
            dataset_dir (str): The directory path of the dataset.
            mix_num (int, optional): The number of audio mixtures. Defaults to 2.
            segment (float, optional): The duration of each audio segment in seconds. Defaults to 2.
            sample_rate (int, optional): The sample rate of the audio. Defaults to 8000.
            mode (str, optional): The mode of the dataset ["clean", "sigle", "both"]. Defaults to "clean".
            dynamic_mix (bool, optional): Whether to generate data dynamically. Defaults to True.
        """
        self.dataset_dir = dataset_dir
        self.sample_rate  = sample_rate
        self.segment = segment
        self.mode = mode
        self.mix_num = mix_num
        self.dynamic_mix = dynamic_mix
        if segment is not None:
            self.segment = int(segment * self.sample_rate)

        if dynamic_mix:
            self._gen_data_list_dynamic()
        else:
            self._gen_data_list_static()

    def _gen_data_list_dynamic(self):

        # TODO: implement dynamic mix
        pass

    def _gen_data_list_static(self):

        mix_dir = os.path.join(self.dataset_dir, f"mix_{self.mode}")
        dataset_list = os.listdir(mix_dir)
        dataset_list.sort()
        self.mix_dataset = AudioDataset(dataset_dir=mix_dir,
                                        dataset_list=dataset_list,
                                        sample_rate=self.sample_rate)
        
        self.gt_datasets = []
        for i in range(self.mix_num):
            gt_dir = os.path.join(self.dataset_dir, f"s{i+1}")
            self.gt_datasets.append(AudioDataset(dataset_dir=gt_dir,
                                                 dataset_list=dataset_list,
                                                 sample_rate=self.sample_rate))
    
    def _getitem_static(self, index):

        mix_audio = self.mix_dataset[index]
        clean_audio = []
        for i in range(self.mix_num):
            clean_audio.append(self.gt_datasets[i][index])
        clean_audio = torch.cat(clean_audio, dim=0)

        sound_len = mix_audio.shape[-1]
        if self.segment is None:
            return mix_audio, clean_audio
        
        if sound_len < self.segment:
            mix_audio = F.pad(mix_audio, (0, self.segment - sound_len))
            clean_audio = F.pad(clean_audio, (0, self.segment - sound_len))
        else:
            offset = random.randint(0, sound_len - self.segment)
            mix_audio = mix_audio[..., offset : offset+self.segment]
            clean_audio = clean_audio[..., offset : offset+self.segment]

        return mix_audio, clean_audio

    def _getitem_dynamic(self, index):
        
        # TODO: implement dynamic mix
        pass

    def __getitem__(self, index):
        if self.dynamic_mix:
            return self._getitem_dynamic(index)
        else:
            return self._getitem_static(index)

    def __len__(self):
        return len(self.mix_dataset)

def collate_fn(batch):
    batch = [x for x in zip(*batch)]
    mix, clean = batch
    return {
        "mix":torch.stack(mix,0),
        "clean":torch.stack(clean,0)}

if __name__ == "__main__":
    dataset = LibriMixDataset(dataset_dir=r"F:\LibriMix\Libri2Mix\wav8k\min\train-360",
                              mix_num=2,
                              segment=2,
                              sample_rate=8000,
                              mode="clean",
                              dynamic_mix=False)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=True,
                                             num_workers=0,
                                             collate_fn=collate_fn)
    for data in dataloader:
        print(data["noisy"].shape)
        print(data["clean"].shape)
        break