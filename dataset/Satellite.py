import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import rasterio


def get_loc_paths(loc_dir:str, ic=None)-> list:
  loc_paths = list()
  for date in sorted(os.listdir(loc_dir)):
    date_path = os.path.join(loc_dir, date)

    if not os.path.isdir(date_path):
      continue
    date_images = list()
    for image in os.listdir(date_path):
      # if specified a single ic & if the name of the image matches the ic desired
      if ic is not None:
          if ic in image:
              date_images.append(os.path.join(date_path, image))
      else:
          # if we want all ics
          date_images.append(os.path.join(date_path,image))
    loc_paths.append(date_images)
  loc_paths = [paths for paths in loc_paths if len(paths)!=0]
  return loc_paths


class SatelliteLoader(Dataset):
    def __init__(self, data_root, is_training , inter_frames=3, n_inputs=4, ic='modis'):
        """
        Creates a Vimeo Septuplet object.
        Inputs.
            data_root: Root path for the Vimeo dataset containing the sep tuples.
            is_training: Train/Test.
            
        """
        super().__init__()
        self.data_root = data_root
        self.training = is_training

        self.inter_frames = inter_frames
        self.n_inputs = n_inputs
        self.set_length = (n_inputs-1)*(inter_frames+1)+1 ## We require these many frames in total for interpolating `interFrames` number of
                                                ## intermediate frames with `n_input` input frames.
        self.paths = get_loc_paths(data_root, ic)

        if self.training:
          self.transforms =  transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomCrop(256)
                #transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                #transforms.ToTensor()
            ])
        
        #else:
        #    self.transforms = transforms.Compose([
        #        transforms.ToTensor()
        #    ])

    def __getitem__(self, index):
        # get the paths corresponding to the images needed from the index
        img_paths = [self.paths[i+index] for i in range(self.set_length)]
        
        # Load images as tensors
        images = list()
        for pth in img_paths: 
          with rasterio.open(pth[0]) as src:
            images.append(torch.from_numpy(src.read(out_dtype='float')[:3]))

        # apply transformations if training
        if self.training:
            seed = random.randint(0, 2**32)
            images_ = []
            for img_ in images:
                # Apply the same transformation by using the same seed
                random.seed(seed)
                images_.append(self.transforms(img_))
            images = images_
            # Random Temporal Flip
            if random.random() >= 0.5:
                images = images[::-1]
        # pick out every inter_frame+1 images as inputs
        inp_images = [images[idx] for idx in range(0, self.set_length, self.inter_frames+1)]   
        rem = self.inter_frames%2
        gt_images = [images[idx] for idx in range(self.set_length//2-self.inter_frames//2 , self.set_length//2+self.inter_frames//2+rem)]  
        return inp_images, gt_images

    def __len__(self):
        if self.training:
            return len(self.paths)-self.set_length+1

def get_loader(data_root, batch_size, shuffle, num_workers, is_training=True, inter_frames=3, n_inputs=4, ic='modis'):
    dataset = SatelliteLoader(data_root , is_training, inter_frames=inter_frames, n_inputs=n_inputs, ic=ic)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


if __name__ == "__main__":
    dataset = SatelliteLoader('/content/drive/MyDrive/las_vegas', is_training=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
