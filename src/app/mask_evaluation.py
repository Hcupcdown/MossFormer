
import torch
import torch.nn as nn
import torchaudio
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data.dataset import *
from model.models import MaskFusionNet, SoundOnlyNet
from utils import *
from utils.metric import *
from utils.stft_loss import *
from utils.time_loss import *


class Tester:
    def __init__(self, args):
        
        self.args        = args
        self.testset     =  TestDataset(args.dataset['test'], sample_rate=args.setting['sample_rate'])
        self.test_loader =  DataLoader(self.testset, batch_size=1, shuffle=False, num_workers=args.num_worker)
        self.criterion = nn.MSELoss()
        print('--- Load vqvae BASE or Large ---')
        self.model = MaskFusionNet(**args.vqvae).to(args.device)   
        
    @staticmethod
    def normal(x):
        std = x.std(dim=-1, keepdim=True)
        x = x / std
        return x, std
    
    @staticmethod
    def radar_normal(x):
        std = x.std(dim = (-1, -2), keepdim=True)
        x = x / std
        return x, std
    
    def test(self, test_type='Test'):
        
        checkpoint = torch.load("log/clean_sigmoid_mask/model/best_val.pth")
        self.model.load_state_dict(checkpoint['state_dict'])
        
        csig = 0
        cbak = 0
        covl = 0
        total_loss = 0    
        sisdr = 0
        total_pesq = 0
        total_stoi = 0
        if test_type == 'Test':
            data_loader = self.test_loader
        else:
            data_loader = self.val_loader
        # self.model.eval()
        with torch.no_grad():
            for i, (noise, clean, radar, mask) in enumerate(tqdm(data_loader)):

                noise, clean = noise.to(self.args.device), clean.to(self.args.device)
                radar = radar.to(self.args.device)
                mask = mask.to(self.args.device)
                clean, std = self.normal(clean)
                noise, std = self.normal(noise)

                estimate = self.model(noise, clean)
                
                loss = self.criterion(clean.squeeze(1), estimate.squeeze(1))
                torchaudio.save("samples/%d.wav"%(i), src=(std*estimate)[0].detach().cpu(), sample_rate = 16000)
                sisdr += torch_sisdr(clean, estimate).item()
                temp_pesq, temp_stoi = get_scores(clean, estimate, self.args)
                total_pesq += temp_pesq
                total_stoi += temp_stoi
                total_loss += loss.item()

            pesq = total_pesq/(i+1)
            stoi = total_stoi/(i+1)
            covl = covl/(i+1)
            cbak = cbak/(i+1)
            csig = csig/(i+1)

            sisdr = sisdr/(i+1)
            print(total_loss/(i+1))
            print("PESQ: {:.4f} |STOI:{:.4f}| covl: {:.4f} | cbak: {:.4f}|csig:{:.4f} | si-sdr:{:.8f}".format(pesq, stoi, covl, cbak, csig, sisdr))             

