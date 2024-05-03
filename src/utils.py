import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms
import torch.utils.data as data
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class ASL_mnist(data.Dataset):
    def __init__(self, dataset_path, transform=None):
        self.df = pd.read_csv(dataset_path+'/sign_mnist_train.csv')
        self.transform = transform
        alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        self.class_to_idx = {alphabet[idx]:idx  for idx in range(26)}
        self.idx_to_class = {idx:alphabet[idx]  for idx in range(26)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        label = self.df.iloc[index, 0]
        image = self.df.iloc[index, 1:].values.reshape(1 ,28, 28)
        image = torch.Tensor(image)
        if self.transform is not None:
            image =self.transform(image)
        return image, label

def show_dataset_images(dataset, transform, num_images=24, start_idx=0):
    data = [dataset[i] for i in range(start_idx, start_idx+num_images)]
    imgs, labels = map(list, zip(*data))
    class_names = {idx: cls for cls, idx in dataset.class_to_idx.items()}
    
    #plt.figure(figsize=(12, 6))
    for i in range(num_images):
      plt.subplot(num_images//6+1, 6, i + 1)
      plt.imshow(transform(imgs[i]))
      plt.title(class_names[int(labels[i])], fontsize=10)
      plt.subplots_adjust(hspace=0.5)
      plt.axis('off') 
                
    plt.gcf().tight_layout()

def numel(m: torch.nn.Module, only_trainable: bool = True):
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)

def show_samples(diffusion_model, reverse_transform, w=0.5):
    images = diffusion_model.sample(w=w)
    for idx, img in enumerate(images):
        plt.subplot(len(images)//6+1, 6, idx+1)
        plt.imshow(reverse_transform(img))
        plt.axis('off') 
    plt.gcf().tight_layout()
    plt.show()





