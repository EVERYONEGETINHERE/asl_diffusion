import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


def LinearBetaSchedule(num_timesteps):
    beta_start = 0.0005
    beta_end = 0.035
    return torch.linspace(beta_start, beta_end, num_timesteps)

def SigmoidBetaSchedule(timesteps):
    beta_start = 0.0005
    beta_end = 0.035
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
    
class Diffusion:
    def __init__(self, denoiser_model, device, num_timesteps=100, schedule=LinearBetaSchedule):
        self.num_timesteps = num_timesteps
        self.betas = schedule(self.num_timesteps).to(device)

        self.alphas = 1. - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, axis=0)

        self.means = torch.sqrt(self.alpha_cumprod)
        self.stds = torch.sqrt(1.- self.alpha_cumprod)
        
        self.model = denoiser_model.to(device)
        self.results = {"loss_history": [], "n_updates": 0}
        self.device = device

    def add_noise(self, original_images, timestep, noise=None):
        if noise is None:
            noise = torch.randn_like(original_images)
        mean = self.means.gather(0, timestep).reshape(-1, 1, 1, 1)
        std = self.stds.gather(0, timestep).reshape(-1, 1, 1, 1)
        noisy_images = mean*original_images + std*noise
        return noisy_images

    @staticmethod
    def diffusion_loss(noise, predicted_noise):
        return F.mse_loss(noise, predicted_noise)

    def diffusion_train_epoch(self, data_loader, lr=1e-3):
        self.model.train()
        train_loss = 0
        loss = None
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        for batch_idx, (data, labels) in enumerate(data_loader):
            optimizer.zero_grad()
            data = data.to(self.device)
            labels = labels.to(self.device)
            timesteps = torch.randint_like(labels, self.num_timesteps)
            if np.random.random() < 0.1:
                labels = None
            noise = torch.randn_like(data)
            noisy_images = self.add_noise(data, timesteps, noise)
            
            predicted_noise = self.model(noisy_images, timesteps, labels)
            loss = self.diffusion_loss(noise, predicted_noise)
            loss.backward()
            optimizer.step()
            self.results['loss_history'].append(loss.cpu().detach().numpy())
            self.results['n_updates'] += 1

    @torch.inference_mode()
    def sample(self, sample_j_and_z = False, return_history=False):
        self.model.eval()
        labels = [i for i in range(26) if (i != 9 and i != 25) or sample_j_and_z]
        
        images = torch.randn([len(labels), 1, 28, 28]).to(self.device)   #initial noise
        labels = torch.Tensor(labels).type(torch.int64).to(self.device)

        for i in range(self.num_timesteps-1, -1, -1):
            timesteps = torch.Tensor([i]).to(self.device).repeat(len(labels))
            predicted_noise = self.model(images, timesteps, labels)
            
            noise = torch.randn_like(images) if i > 1 else 0
            images = (1/torch.sqrt(self.alphas[i]))*(images - predicted_noise*((1-self.alphas[i])/torch.sqrt(1 - self.alpha_cumprod[i]))) + torch.sqrt(self.betas[i])*noise
        return images

    def save_model(self):
        torch.save(self.model.state_dict(), "weights/model_state.pt")


    def load_model(self):
        state_dict = torch.load("weights/model_state.pt")
        self.model.load_state_dict(state_dict)






            
        