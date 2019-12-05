# CSCDNet: Correlated Siamese Change Detection Network (CSCDNet)
This is an official implementation of "Correlated Siamese Change Detection Network (CSCDNet)" in　"[Weakly Supervised Silhouette-based Semantic Change Detection](https://arxiv.org/abs/1811.11985)".

<p align="center">
    <img src='https://drive.google.com/uc?export=view&id=1g0oPp5Kw4chnQ_FSyxc2TNdnNdlz9ZD0' width=95%/></a>
</p>

## Environments
This code was developed and tested with Python 3.6.8 and PyTorch 1.0 and CUDA 9.2.
* GCC
```
# Build and install GCC (>= 7.4.0) if not installed
# Set path variables
export PATH=/home/$USER/local/gcc/bin:$PATH  
export LD_LIBRARY_PATH=/home/$USER/local/gcc/lib64:$LD_LIBRARY_PATH  
```

* Virtualenv for system setting
```
# Set CUDA path. 
# In case of server, the following CUDA path setting with module load command might be necessary.
module load cuda/9.2/9.2.88.1  
 
# Create a virtualenv environment
virtualenv -p python /path/to/env/pytorch1.0cuda9.2 

#Activate the virtualenv environment
source /path/to/env/pytorch1.0cuda9.2/bin/activate

# Install dependencies
pip install -r requirements.txt
```

* Download the pretrained model of resnet18
```
sh download_resnet.sh
```

* Build correlation layer package from [flownet2](https://github.com/NVIDIA/flownet2-pytorch).
```
sh build_correlation_package.sh
```

## Dataset
TSUNAMI and GSV in Panoramic Change Detection dataset are available through an e-mail contact described [here](http://www.vision.is.tohoku.ac.jp/us/research/4d_city_modeling/pano_cd_dataset/) including the dataset used for five-fold cross validation in our paper, in which image cropping and data augumentation have been performed.

Training
```
pcd_5cv        
   ├── set0/                       
   │   ├── train/             # *.jpg
   │   ├── test/              # *.jpg
   │   ├── mask/              # *.png
   |   ├── train.txt
   |   ├── test.txt
   ├── set1/                       
   ...   
   ├── set2/
   ...   
   ├── set3/
   ...
   ├── set4/                       
       ├── train/             # *.jpg
       ├── test/              # *.jpg
       ├── mask/              # *.png
       ├── train.txt
       ├── test.txt   
```

Testing
```
pcd                        
   ├── TSUNAMI/                       
   │   ├── t0/                # *.jpg
   │   ├── t1/                # *.jpg
   │   ├── mask/              # *.png
   ├── GSV/                       
       ├── t0/                # *.jpg
       ├── t1/                # *.jpg
       ├── mask/              # *.png
```


## Training
Train change detection network with correlation layers (CSCDNet)
```
# i-th set of five-hold cross-validation  (0 <= i < 5)
python train.py  --cvset i --use-corr --datadir /path/to/pcd_5cv --checkpointdir /path/to/log --max-iteration 50000 --num-workers 16 --batch-size 32 --icount-plot 50 --icount-save 10000
```

Train change detection network without correlation layers (CDNet)
```
# i-th set of five-hold cross-validation  (0 <= i < 5)
python train.py  --cvset i --datadir /path/to/pcd_5cv --checkpointdir /path/to/log --max-iteration 50000 --num-workers 16 --batch-size 32 --icount-plot 50 --icount-save 10000
```

You can start a tensorboard session
```
tensorboard --logdir=/path/to/log 
```


## Testing
CSCDNet
```
python test.py --use-corr --dataset PCD --datadir /path/to/pcd --checkpointdir /path/to/log/cscdnet/checkpoint
```
CDNet
```
python test.py --dataset PCD --datadir /path/to/pcd --checkpointdir /path/to/log/cdnet/checkpoint
```

## Citation
If you find this implementation useful in your work, please cite the paper. Here is a BibTeX entry:
```
@article{sakurada2018weakly,
  title={Weakly Supervised Silhouette-based Semantic Change Detection},
  author={Sakurada, Ken},
  journal={arXiv preprint arXiv:1811.11985},
  year={2018}
}
```
The preprint can be found [here](https://arxiv.org/abs/1811.11985).
