import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import os

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.model_STVEN = STVEN_Generator().to(self.device)
        self.model_rPPGNet = rPPGNet().to(self.device)
        self.model_fixOri_rPPGNet = rPPGNet().to(self.device)
        
        # Load pretrained weights if available
        if config.MODEL.STVEN_PRETRAINED:
            self.model_STVEN.load_state_dict(torch.load(config.MODEL.STVEN_PRETRAINED))
        if config.MODEL.RPPG_PRETRAINED:
            self.model_rPPGNet.load_state_dict(torch.load(config.MODEL.RPPG_PRETRAINED))
            self.model_fixOri_rPPGNet.load_state_dict(torch.load(config.MODEL.RPPG_PRETRAINED))
        
        # Fix weights for rPPGNet
        for param in self.model_rPPGNet.parameters():
            param.requires_grad = False
        for param in self.model_fixOri_rPPGNet.parameters():
            param.requires_grad = False
            
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model_STVEN.parameters(),
            lr=config.TRAIN.LR_STVEN,
            betas=(config.TRAIN.BETA1, config.TRAIN.BETA2)
        )
        
        # Initialize loss functions
        self.criterion_Binary = nn.BCELoss()
        self.criterion_Pearson = Neg_Pearson()
        
    def train_epoch(self, dataloader, epoch):
        self.model_STVEN.train()
        epoch_loss = 0
        
        with tqdm(dataloader, desc=f'Epoch {epoch}') as pbar:
            for i, batch in enumerate(pbar):
                frames = batch['frames'].to(self.device)  # [B, C, T, H, W]
                bvp = batch['bvp'].to(self.device)       # [B, T]
                
                # Forward STVEN
                enhanced_frames = self.model_STVEN(frames)
                
                # Calculate STVEN losses
                l1_loss = torch.mean(torch.abs(enhanced_frames - frames))
                psnr_loss = self.psnr(enhanced_frames, frames)
                loss_stven = l1_loss + psnr_loss
                
                # Forward enhanced frames through rPPGNet
                skin_map, rppg_aux, rppg, rppg_sa1, rppg_sa2, rppg_sa3, rppg_sa4, visual64, visual32 = \
                    self.model_rPPGNet(enhanced_frames)
                
                # Normalize signals
                rppg = self._normalize_signal(rppg)
                rppg_sa1 = self._normalize_signal(rppg_sa1)
                rppg_sa2 = self._normalize_signal(rppg_sa2)
                rppg_sa3 = self._normalize_signal(rppg_sa3)
                rppg_sa4 = self._normalize_signal(rppg_sa4)
                rppg_aux = self._normalize_signal(rppg_aux)
                
                # Calculate rPPGNet losses
                loss_ecg = self.criterion_Pearson(rppg, bvp)
                loss_ecg_aux = (
                    self.criterion_Pearson(rppg_sa1, bvp) +
                    self.criterion_Pearson(rppg_sa2, bvp) +
                    self.criterion_Pearson(rppg_sa3, bvp) +
                    self.criterion_Pearson(rppg_sa4, bvp) +
                    self.criterion_Pearson(rppg_aux, bvp)
                )
                
                # Calculate perceptual losses
                with torch.no_grad():
                    skin_map_gt, _, rppg_gt, rppg_sa1_gt, rppg_sa2_gt, rppg_sa3_gt, rppg_sa4_gt, visual64_gt, visual32_gt = \
                        self.model_fixOri_rPPGNet(frames)
                
                loss_perceptual = (
                    self.criterion_Pearson(rppg, self._normalize_signal(rppg_gt)) +
                    self.criterion_Pearson(rppg_sa1, self._normalize_signal(rppg_sa1_gt)) +
                    self.criterion_Pearson(rppg_sa2, self._normalize_signal(rppg_sa2_gt)) +
                    self.criterion_Pearson(rppg_sa3, self._normalize_signal(rppg_sa3_gt)) +
                    self.criterion_Pearson(rppg_sa4, self._normalize_signal(rppg_sa4_gt)) +
                    nn.MSELoss()(visual64, visual64_gt) +
                    nn.MSELoss()(visual32, visual32_gt)
                )
                
                # Total loss
                loss = (
                    self.config.LOSS.ECG * loss_ecg +
                    self.config.LOSS.ECG_AUX * loss_ecg_aux +
                    self.config.LOSS.PERCEPTUAL * loss_perceptual +
                    self.config.LOSS.STVEN * loss_stven
                )
                
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                if i % self.config.TRAIN.LOG_INTERVAL == 0:
                    pbar.set_postfix({
                        'loss': loss.item(),
                        'ecg_loss': loss_ecg.item(),
                        'stven_loss': loss_stven.item()
                    })
        
        return epoch_loss / len(dataloader)
    
    @staticmethod
    def _normalize_signal(signal):
        return (signal - torch.mean(signal)) / torch.std(signal)
    
    @staticmethod
    def psnr(img, img_g):
        mse = nn.MSELoss()(img, img_g)
        return 10 * torch.log10(1. / (mse + 1e-8))
    
    def save_checkpoint(self, epoch, loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model_STVEN.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }
        path = f'checkpoints/stven_epoch_{epoch}.pth'
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(checkpoint, path)
