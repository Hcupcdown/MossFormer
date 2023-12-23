import os
import time

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from torch.utils.tensorboard import SummaryWriter


class Log():
    def __init__(self, args) -> None:
        self.log_dir = os.path.join("log", self.generate_filename())
        self.writer = SummaryWriter(
            log_dir=os.path.join(self.log_dir, "TensorBoard"),
            comment="direct skip connect",
        )
        self.model_dir = os.path.join(self.log_dir, "model")
        self.cache_dir = os.path.join(self.log_dir, "cache")
        os.makedirs(self.model_dir)
        os.makedirs(self.cache_dir)

        self.record_config(args)

    def to_spec(self, x):

        plt.specgram(x.detach().cpu().numpy(), NFFT=512, Fs=8000, noverlap=256)
        plt.savefig(os.path.join(self.cache_dir, "spec.png"), dpi=300)
        plt.close()

        spec_img = PIL.Image.open(os.path.join(self.cache_dir, "spec.png"))
        return np.array(spec_img)
    
    def record_config(self, args):
        config_text = os.path.join(self.log_dir, "log.txt")
        with open(config_text, "w") as f:
            f.writelines(repr(args))

    @staticmethod
    def generate_filename():
        return time.strftime("%y-%m-%d-%H-%M-%S", time.localtime())

    def mask_log_dir(self):
        if not os.path.exists(os.path.join(self.log_dir, "temp")):
            os.makedirs(os.path.join(self.log_dir, "temp"))
        
    def add_scalar(self, cate:str, global_step:int, **kwargs):
        for k, v in kwargs.items():
            if torch.is_tensor(v):
                v = v.item()
            self.writer.add_scalar(tag=f"{cate}/{k}",
                                   scalar_value=v,
                                   global_step=global_step)
            
    def add_image(self, cate:str, global_step:int, dataformats:str="CHW", **kwargs):

        for k, v in kwargs.items():
            self.writer.add_image(tag=f"{cate}/{k}_image",
                                  img_tensor=v,
                                  global_step=global_step,
                                  dataformats=dataformats)

    def add_audio(self, cate:str, global_step:int, add_spec:bool = True, **kwargs):

        if add_spec:
            spec_imags = dict()
        for k, v in kwargs.items():
            self.writer.add_audio(tag=f"{cate}/{k}_audio",
                                  snd_tensor=v,
                                  global_step=global_step,
                                  sample_rate=8000)
            if add_spec:
                spec_imags[f"{k}_spec"] = self.to_spec(v[0] if v.dim() > 1 else v)

        if add_spec:
            self.add_image(cate, global_step, "HWC", **spec_imags)
    
    def gen_mask_img(self, x):
        print(x.shape)
        plt.plot(x.detach().cpu().numpy())
        plt.savefig(os.path.join(self.cache_dir, "spec.png"), dpi=300)
        plt.close()

        spec_img = PIL.Image.open(os.path.join(self.cache_dir, "spec.png"))
        return np.array(spec_img)
    
    def add_mask(self, cate:str, global_step:int, **kwargs):
        images = dict()
        for k, v in kwargs.items():
            images[f"{k}_mask"] = self.gen_mask_img(v)
        self.add_image(cate, global_step, "HWC", **images)
        

    def save(self, checkpoint, name):
        torch.save(checkpoint, os.path.join(self.model_dir, name))