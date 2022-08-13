import argparse
import random
import numpy as np
import os

def get_args():
    parser = argparse.ArgumentParser(description="MESSL", formatter_class=argparse.RawTextHelpFormatter)

    # hyperparameters
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--batch-fs", type=int, default=20, help="batch size for few shot runs")
    parser.add_argument("--feature-maps", type=int, default=64, help="number of feature maps")
    parser.add_argument("--preprocessing", type=str, default="",
                        help="preprocessing sequence for few shot, can contain R:relu P:sqrt E:sphering and M:centering")
    parser.add_argument("--postprocessing", type=str, default="",
                        help="postprocessing sequence for few shot, can contain R:relu P:sqrt E:sphering and M:centering")

    parser.add_argument("--device", type=str, default="cuda:0",
                        help="device(s) to use, for multiple GPUs try cuda:ijk, will not work with 10+ GPUs")
    parser.add_argument("--dataset-path", type=str, default='data', help="dataset path")
    parser.add_argument("--dataset-device", type=str, default="",
                        help="use a different device for storing the datasets (use 'cpu' if you are lacking VRAM)")
    parser.add_argument("--dataset", type=str, default="miniImageNet", help="dataset to use")
    parser.add_argument("--test-features", type=str, default="", help="test features and exit")
    parser.add_argument("--seed", type=int, default=-1,
                        help="set random seed manually, and also use deterministic approach")

    parser.add_argument("--n-shots", type=str, default="[1,5]",
                        help="how many shots per few-shot run, can be int or list of ints. In case of episodic training, use first item of list as number of shots.")
    parser.add_argument("--n-runs", type=int, default=10000, help="number of few-shot runs")
    parser.add_argument("--n-ways", type=int, default=5, help="number of few-shot ways")
    parser.add_argument("--n-queries", type=int, default=15, help="number of few-shot queries")
    # transductive
    parser.add_argument("--transductive", action="store_true", help="test features in transductive setting")
    parser.add_argument("--transductive-softkmeans", action="store_true",
                        help="use softkmeans for few-shot transductive")
    parser.add_argument("--transductive-temperature", type=float, default=14,
                        help="temperature for few-shot transductive")
    parser.add_argument("--transductive-temperature-softkmeans", type=float, default=20,
                        help="temperature for few-shot transductive is using softkmeans")

    parser.add_argument('--test-epochs', default=100, type=int, help="the epochs of test")
    parser.add_argument("--logger", type=str, default="", help="save logger file to test")

    args = parser.parse_args()

    if args.dataset_device == "":
        args.dataset_device = args.device

    if args.dataset_path[-1] != '/':
        args.dataset_path += "/"

    if args.device[:5] == "cuda:" and len(args.device) > 5:
        args.devices = []
        for i in range(len(args.device) - 5):
            args.devices.append(int(args.device[i + 5]))
        args.device = args.device[:6]
    else:
        args.devices = [args.device]

    if args.seed == -1:
        args.seed = random.randint(0, 1000000000)

    try:
        n_shots = int(args.n_shots)
        args.n_shots = [n_shots]
    except:
        args.n_shots = eval(args.n_shots)


    return args
