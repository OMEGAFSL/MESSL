from __future__ import print_function
import numpy as np
import torchvision.transforms as transforms



'''
遥感图像预处理  NWPU45
'''
EXP_SIZE = 84

mean_nuwp45 = [0.36801905, 0.3809775, 0.34357441]
std_nupw45 = [0.14530348, 0.13557449, 0.13204114]
normalize_nuwp45 = transforms.Normalize(mean=mean_nuwp45, std=std_nupw45)

transform_Nwpu45 = [
    transforms.Compose([
        transforms.RandomCrop(EXP_SIZE, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize_nuwp45
    ]),

    transforms.Compose([
        transforms.ToTensor(),
        normalize_nuwp45
    ])
]

'''
遥感图像预处理  WHURS19
'''
mean_whurs19 = [0.4259836, 0.44791446, 0.40230706]
std_whurs19 = [0.16213258, 0.14853502, 0.1475367 ]
normalize_whurs19 = transforms.Normalize(mean=mean_whurs19, std=std_whurs19)

transform_Whurs19 = [
    transforms.Compose([
        transforms.RandomCrop(EXP_SIZE, padding=4),
        transforms.RandomHorizontalFlip(),
        lambda x: np.asarray(x).copy(),
        transforms.ToTensor(),
        normalize_whurs19
    ]),

    transforms.Compose([
        transforms.ToTensor(),
        normalize_whurs19
    ])
]

'''
遥感图像预处理  UCM
'''
mean_ucm = [0.48422759, 0.49005176, 0.45050278]
std_ucm = [0.17348298, 0.16352356, 0.15547497]
normalize_ucm = transforms.Normalize(mean=mean_ucm, std=std_ucm)

transform_Ucm = [
    transforms.Compose([
        transforms.RandomCrop(EXP_SIZE, padding=4),
        transforms.RandomHorizontalFlip(),
        lambda x: np.asarray(x).copy(),
        transforms.ToTensor(),
        normalize_ucm
    ]),

    transforms.Compose([
        transforms.ToTensor(),
        normalize_ucm
    ])
]

"""
transorms 列表
"""

transforms_list = ['N', 'W', 'U']

transforms_options = {
    'N': transform_Nwpu45,
    'W': transform_Whurs19,
    'U': transform_Ucm,
}

if __name__ == '__main__':
    print(transforms_options['N'][0])