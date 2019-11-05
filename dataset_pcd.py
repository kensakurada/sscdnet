import numpy as np
import os
import cv2
import torch
from torch.utils.data import Dataset

EXTENSIONS = ['jpg','.png']

def check_img(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def get_img_path(root, basename, extension):
    return os.path.join(root, basename+extension)

def get_img_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class PCD(Dataset):

    def __init__(self, root):
        super(PCD, self).__init__()
        self.img_t0_root = os.path.join(root, 't0')
        self.img_t1_root = os.path.join(root, 't1')
        self.mask_root = os.path.join(root, 'mask')

        self.filenames = [get_img_basename(f) for f in os.listdir(self.mask_root) if check_img(f)]
        self.filenames.sort()

        print('{}:{}'.format(root,len(self.filenames)))

    def __getitem__(self, index):
        filename = self.filenames[index]

        fn_img_t0 = get_img_path(self.img_t0_root, filename, '.jpg')
        fn_img_t1 = get_img_path(self.img_t1_root, filename, '.jpg')
        fn_mask = get_img_path(self.mask_root, filename, '.png')

        if os.path.isfile(fn_img_t0) == False:
            print ('Error: File Not Found: ' + fn_img_t0)
            exit(-1)
        if os.path.isfile(fn_img_t1) == False:
            print ('Error: File Not Found: ' + fn_img_t1)
            exit(-1)
        if os.path.isfile(fn_mask) == False:
            print ('Error: File Not Found: ' + fn_mask)
            exit(-1)

        img_t0 = cv2.imread(fn_img_t0, cv2.IMREAD_COLOR)
        img_t1 = cv2.imread(fn_img_t1, cv2.IMREAD_COLOR)
        mask = cv2.imread(fn_mask, cv2.IMREAD_GRAYSCALE)
        w,h,c = img_t0.shape
        r = 286./min(w,h)
        # resize images so that min(w, h) == 256
        img_t0 = cv2.resize(img_t0, (int(r*w), int(r*h)))
        img_t1 = cv2.resize(img_t1, (int(r*w), int(r*h)))
        mask = cv2.resize(mask, (int(r*w), int(r*h)))[:,:,np.newaxis]

        img_t0_ = np.asarray(img_t0).astype("f").transpose(2, 0, 1) / 128.0 - 1.0
        img_t1_ = np.asarray(img_t1).astype("f").transpose(2, 0, 1) / 128.0 - 1.0
        # black/white inverting
        mask_ = np.asarray(mask>128).astype("int").transpose(2, 0, 1)

        crop_width = 256
        _,h,w = img_t0_.shape
        x_l = np.random.randint(0,w-crop_width)
        x_r = x_l+crop_width
        y_l = np.random.randint(0,h-crop_width)
        y_r = y_l+crop_width

        input_ = torch.from_numpy(np.concatenate((img_t0_[:,y_l:y_r,x_l:x_r], img_t1_[:,y_l:y_r,x_l:x_r]), axis=0))
        mask_ = torch.from_numpy(mask_[:, y_l:y_r, x_l:x_r]).long()

        return input_, mask_

    def __len__(self):
        return len(self.filenames)

    def get_random_index(self):
        index = np.random.randint(0, len(self.filenames))
        return index



class PCD_full(Dataset):

    def __init__(self, root, id_s, id_e, width, height):
        super(PCD_full, self).__init__()
        self.img_t0_root = os.path.join(root, 't0')
        self.img_t1_root = os.path.join(root, 't1')
        self.mask_root = os.path.join(root, 'mask')

        self.filenames = [get_img_basename(f) for f in os.listdir(self.mask_root) if check_img(f)]
        self.filenames.sort()
        self.filenames = self.filenames[id_s:id_e]

        self.width = width
        self.height = height

    def __getitem__(self, index):
        filename = self.filenames[index]

        fn_img_t0 = get_img_path(self.img_t0_root, filename, '.jpg')
        fn_img_t1 = get_img_path(self.img_t1_root, filename, '.jpg')
        fn_mask = get_img_path(self.mask_root, filename, '.png')

        if os.path.isfile(fn_img_t0) == False:
            print ('Error: File Not Found: ' + fn_img_t0)
            exit(-1)
        if os.path.isfile(fn_img_t1) == False:
            print ('Error: File Not Found: ' + fn_img_t1)
            exit(-1)

        if os.path.isfile(fn_mask) == False:
            print ('Error: File Not Found: ' + fn_mask)
            exit(-1)

        img_t0 = cv2.imread(fn_img_t0, cv2.IMREAD_COLOR)
        img_t1 = cv2.imread(fn_img_t1, cv2.IMREAD_COLOR)
        mask = cv2.imread(fn_mask, cv2.IMREAD_GRAYSCALE)
        h,w,c = img_t0.shape
        r = min(w,h)/256
        w_r = int(256*max(w/256,1))
        h_r = int(256*max(h/256,1))
        # resize images so that min(w, h) == 256
        img_t0 = cv2.resize(img_t0, (w_r, h_r))
        img_t1 = cv2.resize(img_t1, (w_r, h_r))
        mask = cv2.resize(mask, (w_r, h_r))[:,:,np.newaxis]

        img_t0_ = np.asarray(img_t0).astype("f").transpose(2, 0, 1) / 128.0 - 1.0
        img_t1_ = np.asarray(img_t1).astype("f").transpose(2, 0, 1) / 128.0 - 1.0
        mask_ = np.asarray(mask>128).astype("int").transpose(2, 0, 1)
        mask_ = torch.from_numpy(mask_).long()

        return img_t0_, img_t1_, mask_, w, h, w_r, h_r

    def __len__(self):
        return len(self.filenames)


