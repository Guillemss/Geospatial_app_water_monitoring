from sklearn.model_selection import train_test_split
import pandas as pd

class Cloud95Dataset(Dataset):
    def __init__(self, rootpath, path2non_empty_csv, pytorch=True):
        super().__init__()
        
        # Only informative patches from non_empty_csv should be used
        # create path dict for all bands and ground truth from it
        
        filenames = pd.read_csv(path2non_empty_csv)["name"]
        self.data = [self.get_path_dict(rootpath, patch) for patch in filenames]

        self.pytorch = pytorch
        
    def get_path_dict(self, rootpath, patch):
        return {'red': rootpath + "/train_red/red_" + patch + ".TIF",
                 'green':rootpath + "/train_green/green_" + patch + ".TIF",
                 'blue': rootpath + "/train_blue/blue_" + patch + ".TIF",
                 'nir': rootpath + "/train_nir/nir_" + patch + ".TIF",
                 'gt': rootpath + "/train_gt/gt_" + patch + ".TIF"}
                                       
    def __len__(self):
        return len(self.data)
     
    def open_as_array(self, idx, invert=False, include_nir=False):

        raw_rgb = np.stack([np.array(Image.open(self.data[idx]['red'])),
                            np.array(Image.open(self.data[idx]['green'])),
                            np.array(Image.open(self.data[idx]['blue'])),
                           ], axis=2)
    
    if include_nir:
            nir = np.expand_dims(np.array(Image.open(self.data[idx]['nir'])), 2)
            raw_rgb = np.concatenate([raw_rgb, nir], axis=2)
    
        if invert:
            raw_rgb = raw_rgb.transpose((2,0,1))
    
        # normalize
        return (raw_rgb / np.iinfo(raw_rgb.dtype).max)
    

    def open_mask(self, idx, add_dims=False):
        
        raw_mask = np.array(Image.open(self.data[idx]['gt']))
        raw_mask = np.where(raw_mask==255, 1, 0)
        
        return np.expand_dims(raw_mask, 0) if add_dims else raw_mask
        # return np.array([raw_mask, 1 - raw_mask])

    def __getitem__(self, idx):
        
        x = torch.tensor(self.open_as_array(idx, invert=self.pytorch, include_nir=True), dtype=torch.float32)
        y = torch.tensor(self.open_mask(idx, add_dims=False), dtype=torch.torch.int64)
        
        return x, y
    
    def open_as_pil(self, idx):
        
        arr = 256*self.open_as_array(idx)
        return Image.fromarray(arr.astype(np.uint8), 'RGB')
    
    def __repr__(self):
        s = 'Dataset class with {} files'.format(self.__len__())

        return s


# change paths here
rootpath = ''
non_empty_csv = ''

data = Cloud95Dataset(rootpath,
                    non_empty_csv95)

len(data)
