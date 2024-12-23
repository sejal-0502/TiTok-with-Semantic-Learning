import os, json
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms as T
from visual_tokenization.taming.data.base import ImagePaths
from tqdm import tqdm
from time import time
import random
import h5py
from PIL import Image
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from torchvision import transforms


class NumpyToTensor:
    def __call__(self, x):
        assert isinstance(x, np.ndarray), f'input must be a numpy array, got {type(x)}'
        assert x.ndim == 3, 'input must be a 3D array'
        return torch.from_numpy(x).permute(2, 0, 1)

class VQGANPreprocess:
    def __call__(self, x):
        assert isinstance(x, torch.Tensor), 'input must be a tensor'
        return x / 127.5 - 1.0

    def inverse(self, x):
        assert isinstance(x, torch.Tensor), 'input must be a tensor'
        return (x + 1.0) * 127.5
    

class ImagePathsMultiframe(ImagePaths):
    def __init__(self, paths, num_frames, random=True, size=None, crop_size=None, random_crop=False, labels=None, scale=False):
        super().__init__(paths, size, crop_size, random_crop, labels, scale)
        self.num_frames = num_frames
        self.random = random

    def __getitem__(self, i):
        # get folder of video i
        frames = self.labels["file_path_"][i]

        # get frames for video i (random or initial)
        start = np.random.randint(0, len(frames) - self.num_frames) if self.random else 0
        frame_paths = [frames[f_i] for f_i in range(start, start+self.num_frames)]
        frames = [self.preprocess_image(f) for f in frame_paths]

        # pack and return
        example = dict()
        example["frames"] = frames
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class CustomMultiFrame(Dataset):
    def __init__(self, size, num_frames, images_list_file, crop_size=None, scale=False):
        super().__init__()

        self.num_frames = num_frames
        # get FIRST frame paths (i.e. for each video, get TOT_FRAMES-NUM_FRAMES)
        with open(images_list_file, 'r') as f:
            paths = json.load(f)
        
        # data: first frame paths
        self.data = []
        for _, paths_list in paths.items():
            self.data.extend(paths_list[:len(paths_list)-num_frames+1]) # we want to handle 1 frame as well
        
        self.dummy_imagepaths = ImagePaths([], size=size, crop_size=crop_size, random_crop=False, scale=scale)
        self.transform = NumpyToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # get all frames for starting frame i
        first_frame_path = self.data[i]
        # split folder and file name
        dirname, first_frame_filename = os.path.split(first_frame_path)
        # get first frame index number from file name of format 'xxxx.jpg'
        first_frame_idx = int(first_frame_filename.split('.')[0])
        # filenames with format '0000.jpg', replace with inclremental index
        frame_paths = [os.path.join(dirname, f"{f_i:04d}.jpg") for f_i in range(first_frame_idx, first_frame_idx+self.num_frames)]
        
        frames = [self.dummy_imagepaths.preprocess_image(f) for f in frame_paths]
        frames = torch.stack([self.transform(f) for f in frames], 1)
        return frames

class MultiHDF5DatasetMultiFrame(Dataset):
    def __init__(self, size, hdf5_paths_file, num_frames, frame_rate_multiplier=1):
        self.size = (size, size) if isinstance(size, int) else size
        self.num_frames = num_frames

        # if data is stored at a higher frame rate than needed, we can skip some frames:
        # we can only reduce frame rate:
        assert frame_rate_multiplier <= 1, 'frame_rate_multiplier must be <= 1'
        # we can only reduce frame rate by integer factor: reciprocal of frame_rate_multiplier must be an integer
        assert 1/frame_rate_multiplier == int(1/frame_rate_multiplier), 'reciprocal of frame_rate_multiplier must be an integer'
        self.frame_interval = int(1/frame_rate_multiplier)
        
        with open(hdf5_paths_file, 'r') as f:
            self.hdf5_paths = f.read().splitlines()

        #self.files = [h5py.File(path, 'r', rdcc_nbytes=1024*1024*1024*8, rdcc_nslots=1000) for path in self.hdf5_paths]
        self.files = [h5py.File(path, 'r') for path in self.hdf5_paths]
        self.lengths = []
        self.file_keys = []
        for file in self.files:
            keys = [k for k in file.keys() if 'meta_data' not in k]
            self.file_keys.append(keys)
            self.lengths.append({key: len(file[key]) for key in keys})

        self.total_length = sum(sum(lengths.values()) for lengths in self.lengths)
        print(f'Total length: {self.total_length}')
        self.transform = transforms.Compose([transforms.Resize(min(self.size)),
                                         transforms.CenterCrop(self.size),
                                         transforms.ToTensor(),
                                         ])
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        file_index = random.randint(0, len(self.files) - 1)
        h5_file = h5py.File(self.hdf5_paths[file_index], 'r', rdcc_nbytes=1024*1024*1024*400, rdcc_nslots=1000)
        try:
            key_index = random.randint(0, len(self.file_keys[file_index]) - 1)
            # img_start_index = random.randint(0, self.lengths[file_index][self.file_keys[file_index][key_index]] - (self.num_frames+1)*self.frame_interval)
            video_length = self.lengths[file_index][self.file_keys[file_index][key_index]]
            frames_needed_after = (self.num_frames+1)*self.frame_interval
            assert video_length >= frames_needed_after, f'file_index: {file_index}, key_index: {key_index}, video length: {video_length}, frames_needed_after: {frames_needed_after}'
            img_start_index = random.randint(0, video_length - frames_needed_after)
            key = self.file_keys[file_index][key_index]
            if 'meta_data' in key:
                key = key.replace('_meta_data', '')
            #data = self.files[file_index][key][img_start_index:img_start_index+self.num_frames]
            #data = h5_file[key][img_start_index:img_start_index+self.num_frames]
            #images = [Image.fromarray(d) for d in data]
            #images = [self.transform(i) for i in images] # ensure that it is same transform for all images
            #images = [self.transform(Image.fromarray(d)) for d in data]
            
            # images = [self.transform(Image.fromarray(h5_file[key][img_start_index+i]))*2-1 for i in range(self.num_frames)]
            images = [self.transform(Image.fromarray(h5_file[key][img_start_index+i*self.frame_interval]))*2-1 for i in range(self.num_frames)]
        except Exception as e:
            print(e)
            # try again
            return self.__getitem__(idx)
        #images = [image*2-1 for image in images]
        return torch.stack(images, dim=0)
    
    def close(self):
        for file in self.files:
            file.close()


class MultiHDF5DatasetMultiFrameIdxMapping(Dataset):
    '''
    This dataset maps each index to a specific frame in a specific video. Useful for validation and selecting subsets of frames.
    num_frames: number of frames to return for each index

    '''
    def __init__(self, size, hdf5_paths_file, 
                 num_frames, get_every_nth_frame=1,
                 frame_rate_multiplier=1):

        if frame_rate_multiplier != 1:
            raise NotImplementedError
        
        self.size = size
        self.num_frames = num_frames
        self.get_every_nth_frame = get_every_nth_frame
        # expand environment variables in path
        with open(os.path.expandvars(hdf5_paths_file), 'r') as f:
            self.hdf5_paths = f.read().splitlines()

        self.hdf5_files = [h5py.File(path, 'r') for path in self.hdf5_paths]
        self.index_to_starting_frame_map = []
        for file in self.hdf5_files:
            keys = list(file.keys())
            for key in keys:
                video_length = len(file[key])
                # we take every nth frame, as long as we can get num_frames frames after that
                max_frame_index = video_length - num_frames - 1
                for i in range(0, max_frame_index + 1, get_every_nth_frame):
                    self.index_to_starting_frame_map.append((file, key, i))

        print(f'Total length: {len(self.index_to_starting_frame_map)}')
        
        self.transform = transforms.Compose([transforms.Resize(self.size),
                                         transforms.CenterCrop((self.size, self.size)),
                                         transforms.ToTensor(),
                                         ])
        
    def __len__(self):
        return len(self.index_to_starting_frame_map)
    
    def __getitem__(self, idx):
        file, key, start_frame = self.index_to_starting_frame_map[idx]
        images = [self.transform(Image.fromarray(file[key][start_frame+i]))*2-1 for i in range(self.num_frames)]
        return torch.stack(images, dim=0)
    
    def get_sample_and_location(self, idx):
        file, key, start_frame = self.index_to_starting_frame_map[idx]
        images = [self.transform(Image.fromarray(file[key][start_frame+i]))*2-1 for i in range(self.num_frames)]
        return torch.stack(images, dim=0), (file.filename, key, start_frame)

    def close(self):
        for file in self.hdf5_files:
            file.close()


class MultiHDF5DatasetMultiFrameFromJSON(MultiHDF5DatasetMultiFrameIdxMapping):
    """
    The structure of the JSON file should be as follows:
    [
        {
            "h5_path": <PATH TO THE H5 FILE CONTAINING THE VIDEO>, 
            "video_key": <KEY/NAME OF THE VIDEO, e.g. 53773fdf-311fd624>
            "start_frame": <STARTING FRAME INDEX>
        }
    ]
    """
    def __init__(self, size, samples_json, num_frames, frame_rate_multiplier=1):

        if frame_rate_multiplier != 1:
            raise NotImplementedError
        
        self.samples_json = samples_json
        self.size = (size, size) if isinstance(size, int) else size
        self.num_frames = num_frames
        
        # read json
        with open(os.path.expandvars(samples_json), 'r') as f:
            self.samples = json.load(f)
        
        # get all h5 file paths
        h5_paths = list(set([sample['h5_path'] for sample in self.samples]))
        self.hdf5_files = {h5_path: h5py.File(h5_path, 'r') for h5_path in h5_paths}
        self.index_to_starting_frame_map = []
        for sample in self.samples:
            file = self.hdf5_files[sample['h5_path']]
            key = sample['video_key']
            start_frame = sample['start_frame']
            self.index_to_starting_frame_map.append((file, key, start_frame))

        self.transform = transforms.Compose([transforms.Resize(min(self.size)),
                                         transforms.CenterCrop(self.size),
                                         transforms.ToTensor(),
                                         ])
    
    def close(self):
        for file in self.hdf5_files.values():
            file.close()

