class H5Dataset(Dataset):
    def __init__(self, size, hdf5_paths_file):
        # Load the paths to the HDF5 files
        with open(hdf5_paths_file, 'r') as f:
            self.hdf5_paths = f.read().splitlines()
        #self.hdf5_paths = hdf5_paths
        #print(f'Loading {self.hdf5_paths}')
        self.current_h5_file = None
        self.subdirs = []
        self.load_hdf5_file(0)
        self.size = size
        self.transform = transforms.Compose([transforms.Resize(self.size),
                                         transforms.CenterCrop((self.size, self.size)),
                                         transforms.ToTensor(),
                                         ])


    def load_hdf5_file(self, index):
        if self.current_h5_file is not None:
            self.current_h5_file.close()
        print(f'Loading {self.hdf5_paths[index]}')
        self.current_h5_file = h5py.File(f'{self.hdf5_paths[index]}', 'r') # Open the HDF5 file
        self.all_img_files = list(self.current_h5_file.keys())
        print(f'Loaded {self.hdf5_paths[index]} with {len(self.all_img_files)} videos.')

    def __len__(self):
        # Assuming each subdir has 9 images and each file has 4080 subdirs
        return len(self.all_img_files)*1000 # total images 

    def __getitem__(self, idx):
        #print(f'idx: {idx}')
        video_idx = random.randint(0, len(self.all_img_files)-1)
        subdir = self.all_img_files[video_idx]#[img_idx]

        # Random start within the first 46 images to ensure 4 consecutive images
        image_data = self.current_h5_file[subdir] #[:]
        img_idx = random.randint(0, len(image_data) - 1)
        # sample one image from image_data list
        image = Image.fromarray(image_data[img_idx])
        image = self.transform(image)
        return image

    def shuffle_files(self):
        random.shuffle(self.hdf5_paths)
        self.load_hdf5_file(0)  # Reload the first file in the new order


class H5DatasetMany(H5Dataset):
    def __init__(self, size, hdf5_paths_file, num_h5):
        with open(hdf5_paths_file, 'r') as f:
            self.hdf5_paths = f.read().splitlines()

        self.current_h5_file = {}
        self.all_img_files = []
        
        self.size = size
        self.transform = transforms.Compose([transforms.Resize(self.size),
                                         transforms.CenterCrop((self.size, self.size)),
                                         transforms.ToTensor(),
                                         ])
        self.num_h5 = num_h5
        self.load_hdf5_file()
        #self.load_hdf5_file(num_h5=self.num_h5)
       
        #self.total_num_files = self.count_files()
        #print(f'count_files: {self.total_num_files} in {self.num_h5} files')

    def __len__(self):
        # Assuming each subdir has 9 images and each file has 4080 subdirs
        return 10000 #self.total_num_files # total images

    def __getitem__(self, idx):
        #print(f'idx: {idx}')
        video_idx = random.randint(0, len(self.all_img_files)-1)
        subdir = self.all_img_files[video_idx]#[img_idx]

        # Random start within the first 46 images to ensure 4 consecutive images
        image_data = self.current_h5_file[subdir] #[:]
        img_idx = random.randint(0, image_data.shape[0] - 1)
        # sample one image from image_data list
        image = Image.fromarray(image_data[img_idx])
        image = self.transform(image)
        return image

    def load_hdf5_file(self):
        h5_files_paths = random.sample(self.hdf5_paths, self.num_h5)

        # if len(self.current_h5_file) > 0:
        #     for i in range(self.num_h5):
        #         self.current_h5_file[i].close()

        print(h5_files_paths)
        #for i in range(self.num_h5):
        #    self.current_h5_file.update(h5py.File(f'{h5_files_paths[i]}', 'r')) # Open the HDF5 file
        self.current_h5_file = h5py.File(f'{h5_files_paths[0]}', 'r')
            
        self.all_img_files = []
        #for i in range(self.num_h5):
        #    self.all_img_files.extend(list(self.current_h5_file[i].keys()))
        self.all_img_files = list(self.current_h5_file.keys())
        print(f'Loaded {self.num_h5} files with {len(self.all_img_files)} videos.')
        # convert all values to numpy arrays
        for subdir in self.all_img_files:
            self.current_h5_file[subdir] = np.array(self.current_h5_file[subdir])


    def shuffle_files(self):
        random.shuffle(self.hdf5_paths)
        self.load_hdf5_file()  # Reload the first file in the new order


class H5DatasetManyV2(H5Dataset):
    def __init__(self, size, hdf5_paths_file, num_h5):
        with open(hdf5_paths_file, 'r') as f:
            self.hdf5_paths = f.read().splitlines()

        self.current_h5_file = {}
        self.all_img_files = []
        
        self.size = size
        self.transform = transforms.Compose([transforms.Resize(self.size),
                                         transforms.CenterCrop((self.size, self.size)),
                                         transforms.ToTensor(),
                                         ])
        self.num_h5 = num_h5
        #self.load_hdf5_file()
        #self.load_hdf5_file(num_h5=self.num_h5)
       
        self.total_num_files = self.count_files()
        print(f'count_files: {self.total_num_files} in {self.num_h5} h5 files')

    def __len__(self):
        # Assuming each subdir has 9 images and each file has 4080 subdirs
        return 10000 #self.total_num_files # total images

    def count_files(self):
        # h5_files_paths = random.sample(self.hdf5_paths, self.num_h5)
        # for i in range(self.num_h5):
        #     self.current_h5_file.update(h5py.File(f'{h5_files_paths[i]}', 'r')) # Open the HDF5 file
        count = 0
        self.current_h5_file = h5py.File(f'{self.hdf5_paths[0]}', 'r')
        self.all_img_files = list(self.current_h5_file.keys())
        #import pdb; pdb.set_trace()
        for subdir in self.all_img_files:
            image_data = self.current_h5_file[subdir]#[:]
            count += len(image_data)
        return count
    
    def __getitem__(self, idx):
        #h5_files_paths = random.sample(self.hdf5_paths, self.num_h5)
        self.current_h5_file = h5py.File(f'{self.hdf5_paths[0]}', 'r')
        self.all_img_files = list(self.current_h5_file.keys())

        #print(f'idx: {idx}')
        video_idx = random.randint(0, len(self.all_img_files)-1)
        subdir = self.all_img_files[video_idx]#[img_idx]

        # Random start within the first 46 images to ensure 4 consecutive images
        image_data = self.current_h5_file[subdir] #[:]
        img_idx = random.randint(0, image_data.shape[0] - 1)
        # sample one image from image_data list
        image = Image.fromarray(image_data[img_idx])
        image = self.transform(image)
        return image

    def load_hdf5_file(self):
        h5_files_paths = random.sample(self.hdf5_paths, self.num_h5)

        if len(self.current_h5_file) > 0:
            for i in range(self.num_h5):
                self.current_h5_file[i].close()

        # print(h5_files_paths)
        #for i in range(self.num_h5):
        #   self.current_h5_file.update(h5py.File(f'{h5_files_paths[i]}', 'r')) # Open the HDF5 file
        self.current_h5_file = h5py.File(f'{h5_files_paths[0]}', 'r')
            
        self.all_img_files = []
        #for i in range(self.num_h5):
        #   self.all_img_files.extend(list(self.current_h5_file[i].keys()))
        self.all_img_files = list(self.current_h5_file.keys())
        print(f'Loaded {self.num_h5} files with {len(self.all_img_files)} videos.')
        # convert all values to numpy arrays
        # for subdir in self.all_img_files:
        #     self.current_h5_file[subdir] = np.array(self.current_h5_file[subdir])


    def shuffle_files(self):
        random.shuffle(self.hdf5_paths)
        self.load_hdf5_file()  # Reload the first file in the new order


class H5DatasetManyV2Video(H5Dataset):
    def __init__(self, size, hdf5_paths_file, num_h5):
        with open(hdf5_paths_file, 'r') as f:
            self.hdf5_paths = f.read().splitlines()

        self.current_h5_file = {}
        self.all_img_files = []
        self.all_images = {}
        
        
        self.size = size
        self.transform = transforms.Compose([transforms.Resize(self.size),
                                         transforms.CenterCrop((self.size, self.size)),
                                         transforms.ToTensor(),
                                         ])
        self.num_h5 = num_h5
        self.counter = 0
       
        self.total_num_files = self.count_files()
        print(f'count_files: {self.total_num_files} in {self.num_h5} h5 files')

    def __len__(self):
        # Assuming each subdir has 9 images and each file has 4080 subdirs
        return self.total_num_files # total images

    def count_files(self):
        # h5_files_paths = random.sample(self.hdf5_paths, self.num_h5)
        # for i in range(self.num_h5):
        #     self.current_h5_file.update(h5py.File(f'{h5_files_paths[i]}', 'r')) # Open the HDF5 file
        count = 0
        current_h5_file = h5py.File(f'{self.hdf5_paths[0]}', 'r')
        all_img_files = list(current_h5_file.keys())
        all_img_files = [x for x in all_img_files if 'meta_data' not in x]
        
        #import pdb; pdb.set_trace()
        for subdir in all_img_files:
            image_data = current_h5_file[subdir]#[:]
            count += len(image_data)
        return count
    
    def __getitem__(self, idx):
        #h5_files_paths = random.sample(self.hdf5_paths, self.num_h5)
        # for i in range(self.num_h5):
        #     self.current_h5_file.update(h5py.File(f'{self.hdf5_paths[i]}', 'r')) # Open the HDF5 file
        self.counter += 1
        self.current_h5_file = h5py.File(f'{self.hdf5_paths[0]}', 'r')
        self.all_img_files = list(self.current_h5_file.keys())
        #print(self.all_img_files)
        #self.all_img_files = [x for x in self.all_img_files if 'meta_data' not in x]

        #print(f'idx: {idx}')
        video_idx = random.randint(0, len(self.all_img_files)-1)
        subdir = self.all_img_files[video_idx]#[img_idx]
        if 'meta_data' in subdir:
            subdir = subdir.replace('_meta_data', '')

        image_data = self.current_h5_file[subdir]#[:]
        #img_idx = random.randint(0, len(self.current_h5_file[subdir]) - 1)
        img_idx = random.randint(0, image_data.shape[0] - 1)
        # sample one image from image_data list
        image = Image.fromarray(image_data[img_idx])
        image = self.transform(image)

        return image

    def shuffle_files(self):
        random.shuffle(self.hdf5_paths)
        #self.load_hdf5_file()  # Reload the first file in the new order