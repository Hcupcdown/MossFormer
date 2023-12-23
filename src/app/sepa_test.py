
import torch
import torchaudio
from einops import rearrange
from tqdm import tqdm

from data.dataset import *
from utils import *
from utils.metric import *


class FreSepaTester:


    def __init__(self, model, dataloader, n_fft, hop, args):
        
        self.args = args
        self.device = args.device
        self.n_fft = n_fft
        self.hop = hop
        self.model = model.to(args.device)
        self.test_loader = dataloader
        self.loss_weights = [0.1,0.9,0.9,0.2]

    def process_data(self, data):

        noisy = data[0].to(self.args.device)
        clean = data[1].to(self.args.device)
        radar = data[2].to(self.args.device)
        clean = clean.view(-1, clean.size(-1))
        radar = radar.view(radar.size(0)*radar.size(1), 1, *radar.shape[2:])
        noisy = noisy.squeeze(1)

        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
        clean_c = c.repeat_interleave(clean.size(0)//noisy.size(0))
        noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)
        
        noisy, clean = torch.transpose(noisy * c, 0, 1), torch.transpose(clean * clean_c, 0, 1)

        noisy_spec = torch.stft(
            noisy,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(self.device),
            onesided=True,
            return_complex=False
        )
        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)

        clean_spec = torch.stft(
            clean,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(self.device),
            onesided=True,
            return_complex=False
        )
        clean_spec = power_compress(clean_spec).permute(0, 1, 3, 2)
        
        return {
            "clean_audio":clean,
            "noisy_audio":noisy,
            "noisy_spec":noisy_spec,
            "clean_spec":clean_spec,
            "radar":radar,
            "c":c
        }
    
    def post_process_data(self, est_real, est_imag, clean_spec):
        
        clean_spec = clean_spec.permute(0, 1, 3, 2)
        clean_real = clean_spec[:, 0, :, :].unsqueeze(1)
        clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)
        clean_mag = torch.sqrt(clean_real**2 + clean_imag**2)

        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
        est_mag = torch.sqrt(est_real**2 + est_imag**2)
        
        est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
        est_audio = torch.istft(
            est_spec_uncompress,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(self.device),
            onesided=True,
        )
        
        return {
            "est_real": est_real,
            "est_imag": est_imag,
            "est_mag": est_mag,
            "clean_real": clean_real,
            "clean_imag": clean_imag,
            "clean_mag": clean_mag,
            "est_audio": est_audio,
        }
    
    def data2stft(self, data):

        noisy = data["noisy"].to(self.args.device)
        clean = data["clean"].to(self.args.device)

        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
        noisy = noisy * c[:, None]
        if noisy.dim() == clean.dim():
            clean = clean * c[:, None]
        else:
            clean = clean * c[:, None, None]
        
        """[b, c, t] -> [b, c, 2, n_fft, fft_t]"""
        noisy_spec = sound2stft(noisy, self.n_fft, self.hop)
        clean_spec = sound2stft(clean, self.n_fft, self.hop)
        
        return {
            "clean_audio":clean,
            "noisy_audio":noisy,
            "noisy_spec":noisy_spec,
            "clean_spec":clean_spec,
            "c":c
        }
    
    def calculate_loss(self, est_real, est_imag, clean_spec, clean_audio):
        est_real = est_real.permute(0, 1, 3, 2)
        est_imag = est_imag.permute(0, 1, 3, 2)

        clean_real = clean_spec[:, :, 0, :, :]
        clean_imag = clean_spec[:, :, 1, :, :]
        
        est_mag = torch.sqrt(est_real**2 + est_imag**2)
        clean_mag = torch.sqrt(clean_real**2 + clean_imag**2)
        
        est_audio = stft2sound(est_real, est_imag, self.n_fft, self.hop)
        
        loss_mag = F.mse_loss(est_mag, clean_mag)
        loss_ri = F.mse_loss(est_real, clean_real) + F.mse_loss(est_imag, clean_imag)
        min_len = min(est_audio.size(-1), clean_audio.size(-1))
        time_loss = torch.mean(torch.abs(est_audio[...,:min_len] - clean_audio[...,:min_len]))

        loss = (
            self.loss_weights[0] * loss_ri
            + self.loss_weights[1] * loss_mag
            + self.loss_weights[3] * time_loss
        )

        return loss, est_audio
    
    def process_data(self, batch_data):
        sound_data = self.data2stft(batch_data)
        radar_data = batch_data["radar"].to(self.args.device)
        radar_data = rearrange(radar_data, "b c h w -> (b c) h w")
        radar_data = radar_data.unsqueeze(1)
        sound_data["radar"] = radar_data
        return sound_data
    
    def save_wav(self, save_path, audio):
        audio = audio.squeeze(0)
        os.makedirs(os.path.join("result", save_path), exist_ok=True)
        for i in range(audio.shape[0]):
            torchaudio.save(os.path.join("result", save_path,f"{i}.wav"), audio[i].unsqueeze(0).cpu(), 16000)
    
    def test_batch(self, batch_data):
        data = self.process_data(batch_data)
        noisy_in = data["noisy_spec"].permute(0, 1, 3, 2)
        radar = data["radar"]
        est_real, est_imag = self.model(noisy_in, radar)

        loss, est_audio = self.calculate_loss(est_real, est_imag, data["clean_spec"], data["clean_audio"])

        c = data["c"]
        est_audio = est_audio / c[:, None]

        return loss, est_audio

    def test(self):

        total_loss = 0
        sisdr = 0
        total_sisdr = 0
        self.model.eval()
        with torch.no_grad():
            for i, batch_data in enumerate(tqdm(self.test_loader)):
                loss, est_audio = self.test_batch(batch_data)
                total_loss += loss.item()
                self.save_wav(f"{i}", est_audio)
        print("total loss: ", total_loss/(i+1))
        print("total sisdr: ", total_sisdr/(i+1))