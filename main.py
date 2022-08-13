import torch
import random
import numpy as np
from args import get_args
from eval import define_runs, evaluate_shot
from dataset import get_dataset

def main():
    args = get_args()
    # 检查参数
    assert (args.n_runs % args.batch_fs == 0)
    assert args.test_features != "", "Feature Files Error"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 多次测试，保存最好的和最差的测试结果
    best_result = {"test_acc": 0., "test_conf": None}
    worst_result = {"test_acc": 100., "test_conf": None}

    print("Pytorch version: " + torch.__version__)

    # 获取features文件的路径
    try:
        filenames = eval(args.test_features)
    except:
        filenames = args.test_features
    if isinstance(filenames, str):
        filenames = [filenames]

    for e in range(args.test_epochs): # test_epochs默认为100
        # concat features文件
        test_features = torch.cat(
            [torch.load(fn, map_location=torch.device(args.device)).to(args.dataset_device) for fn in filenames], dim=2)
        # 获取数据集及其信息
        # few_shot：是否小样本，True， top_5：是否计算top5准确率，False
        # input_shape：图像尺寸，(3, 84, 84)， num_classes：类别划分信息，(训练、验证、测试、每类数据量）
        loaders, input_shape, num_classes, few_shot, top_5 = get_dataset(args)
        # 数据集初始化
        if few_shot:
            num_classes, val_classes, novel_classes, elements_per_class = num_classes
            elements_val, elements_novel = [elements_per_class] * val_classes, [elements_per_class] * novel_classes
            elements_train = None

            val_runs = list(
                zip(*[define_runs(args.n_ways, s, args.n_queries, val_classes, elements_val, args.n_runs, args.device)
                      for s in args.n_shots]))
            val_run_classes, val_run_indices = val_runs[0], val_runs[1]
            novel_runs = list(
                zip(*[
                    define_runs(args.n_ways, s, args.n_queries, novel_classes, elements_novel, args.n_runs, args.device)
                    for s in args.n_shots]))
            novel_run_classes, novel_run_indices = novel_runs[0], novel_runs[1]
            print("done.")
            few_shot_meta_data = {
                "elements_train": elements_train,
                "val_run_classes": val_run_classes,
                "val_run_indices": val_run_indices,
                "novel_run_classes": novel_run_classes,
                "novel_run_indices": novel_run_indices,
                "best_val_acc": [0] * len(args.n_shots),
                "best_val_acc_ever": [0] * len(args.n_shots),
                "best_novel_acc": [0] * len(args.n_shots)
            }

        train_features = test_features[:num_classes]
        val_features = test_features[num_classes:num_classes + val_classes]
        test_features = test_features[num_classes + val_classes:]
        if not args.transductive:
            for i in range(len(args.n_shots)):
                val_acc, val_conf, test_acc, test_conf = evaluate_shot(args, i, train_features, val_features,
                                                                       test_features, few_shot_meta_data)
                print("Epoch[{}] Inductive {:d}-shot: {:.2f}% (± {:.2f}%)".format(e, args.n_shots[i], 100 * test_acc,
                                                                        100 * test_conf))
        else:
            for i in range(len(args.n_shots)):
                val_acc, val_conf, test_acc, test_conf = evaluate_shot(args, i, train_features, val_features,
                                                                       test_features, few_shot_meta_data,
                                                                       transductive=True)
                print("Epoch[{}] Transductive {:d}-shot: {:.2f}% (± {:.2f}%)".format(e, args.n_shots[i], 100 * test_acc,
                                                                           100 * test_conf))
        if test_acc > best_result["test_acc"]:
            best_result["test_acc"] = test_acc
            best_result["test_conf"] = test_conf

        if test_acc < worst_result["test_acc"]:
            worst_result["test_acc"] = test_acc
            worst_result["test_conf"] = test_conf

    # 打印最优和最差的结果
    print(" The best test result: ")
    print("{:d}-shot: {:.2f}% (± {:.2f}%)".format(args.n_shots[i], 100 * best_result["test_acc"],
                                                  100 * best_result["test_conf"]))
    print(" The worst test result: ")
    print("{:d}-shot: {:.2f}% (± {:.2f}%)".format(args.n_shots[i], 100 * worst_result["test_acc"],
                                                  100 * worst_result["test_conf"]))
    # 汇总所有结果的最优和最劣
    if args.logger != "":
        with open(args.ablation_logger, 'a+') as logger_file:
            logger_file.write("The best test result:")
            logger_file.write("{:d}-shot: {:.2f}% (± {:.2f}%)\n".format(args.n_shots[i], 100 * best_result["test_acc"],
                                                                      100 * best_result["test_conf"]))
            logger_file.write("The worst test result:")
            logger_file.write("{:d}-shot: {:.2f}% (± {:.2f}%)\n\n".format(args.n_shots[i], 100 * worst_result["test_acc"],
                                                                      100 * worst_result["test_conf"]))


if __name__ == '__main__':
    main()






