# MESSL
### Multiform Ensemble Self-Supervised Learning for Few-Shot Remote Sensing Scene Classification

This repository contains the official implementation of Multiform Ensemble Self-Supervised Learning for Few-Shot Remote Sensing Scene Classification (MES2L).

You can use this repository for testing, and we will sort out and upload the rest of the code.

### Requirements

This repo requires the following:

```
numpy==1.19.2
pandas==1.1.5
Pillow==9.2.0
scipy==1.6.2
torch==1.8.0
torchvision==0.9.0
```

you can run `pip3 install -r requirements.txt` to install all the packages. 

### Datasets

For the testing experiments,we used 3 public dataset as following,NWPU RESISC45,UC Merced and WHU-RS19.And we divided these datasets into 3 subsets, training, validation and test set,the number of categories for subsets of each dataset is as follows:

##### NWPU RESISC45

train：25，Val：10，Test：10

##### UC Merced

train：10，Val：5，Test：6

##### WHU-RS19

train：9，Val：5，Test：5

You can download the dataset at this [Google Driver link](https://drive.google.com/drive/folders/1bXaFhQzsNPr-qJ5EkZH-C2eh1RZsIlDm?usp=sharing), then you need to put it in the root directory of the project, and then run the following script to get the corresponding CSV file.

```shell
python create_rsdata_labels.py
```

### Model Files

You can download the corresponding model file and feature file through this [google driver link](https://drive.google.com/drive/folders/10OT6xx66c0V-mj2DJh3RxDWij35aQCL_?usp=sharing). The name of the file indicates the type of the file:

- `mes2l_features.pt1{$k}`: feature file of k shot test(mes2l)
- `ce_features.pt11`:feature file of 1 shot test(Only cross-entropy loss function)

### Test

You can run the following script to test:

- **Inductive**

```shell
python main.py --dataset UCM --test-features "['model_file/UCM/mes2l_features1.pt11']" --preprocessing ME --n-shots 1
```

- **Transductive**

```shell
python main.py --dataset UCM --test-features "['model_file/UCM/mes2l_features1.pt11']" --postprocessing ME  --transductive --transductive-softkmeans --transductive-temperature-softkmeans 5 --n-shots 1
```

