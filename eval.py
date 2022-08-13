import torch
import torch.nn.functional as F
import utils


def define_runs(n_ways, n_shots, n_queries, num_classes, elements_per_class, n_runs, device):
    run_classes = torch.LongTensor(n_runs, n_ways).to(device)
    run_indices = torch.LongTensor(n_runs, n_ways, n_shots + n_queries).to(device)
    for i in range(n_runs):
        run_classes[i] = torch.randperm(num_classes)[:n_ways] # 随机选n_ways类
        for j in range(n_ways):
            run_indices[i,j] = torch.randperm(elements_per_class[run_classes[i, j]])[:n_shots + n_queries] # 随机选数据
    return run_classes, run_indices

def generate_runs(data, run_classes, run_indices, batch_idx, batch_fs):
    n_runs, n_ways, n_samples = run_classes.shape[0], run_classes.shape[1], run_indices.shape[2]
    run_classes = run_classes[batch_idx * batch_fs : (batch_idx + 1) * batch_fs]
    run_indices = run_indices[batch_idx * batch_fs : (batch_idx + 1) * batch_fs]
    run_classes = run_classes.unsqueeze(2).unsqueeze(3).repeat(1,1,data.shape[1], data.shape[2])
    run_indices = run_indices.unsqueeze(3).repeat(1, 1, 1, data.shape[2])
    datas = data.unsqueeze(0).repeat(batch_fs, 1, 1, 1)
    cclasses = torch.gather(datas, 1, run_classes)
    res = torch.gather(cclasses, 2, run_indices)

    return res


def ncm(args, train_features, features, run_classes, run_indices, n_shots, elements_train=None):
    with torch.no_grad():
        dim = features.shape[2]
        targets = torch.arange(args.n_ways).unsqueeze(1).unsqueeze(0).to(args.device)
        features = utils.preprocess(train_features, features, args.preprocessing,elements_train=elements_train)
        scores = []
        for batch_idx in range(args.n_runs // args.batch_fs):
            runs = generate_runs(features, run_classes, run_indices, batch_idx, args.batch_fs)
            means = torch.mean(runs[:,:,:n_shots], dim = 2) # 求平均值
            distances = torch.norm(runs[:,:,n_shots:].reshape(args.batch_fs, args.n_ways, 1, -1, dim) - means.reshape(args.batch_fs, 1, args.n_ways, 1, dim), dim = 4, p = 2)
            winners = torch.min(distances, dim = 2)[1]
            scores += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())
        return utils.stats(scores, "")


def transductive_ncm(args, train_features, features, run_classes, run_indices, n_shots, n_iter_trans, n_iter_trans_sinkhorn, temp_trans, alpha_trans, cosine, elements_train=None):
    with torch.no_grad():
        dim = features.shape[2]
        targets = torch.arange(args.n_ways).unsqueeze(1).unsqueeze(0).to(args.device)
        features = utils.preprocess(train_features, features, args.preprocessing, elements_train=elements_train)
        if cosine:
            features = features / torch.norm(features, dim = 2, keepdim = True)
        scores = []
        for batch_idx in range(args.n_runs // args.batch_fs):
            runs = generate_runs(features, run_classes, run_indices, batch_idx, args.batch_fs)
            means = torch.mean(runs[:,:,:n_shots], dim = 2)
            if cosine:
                means = means / torch.norm(means, dim = 2, keepdim = True)
            for _ in range(n_iter_trans):
                if cosine:
                    similarities = torch.einsum("bswd,bswd->bsw", runs[:,:,n_shots:].reshape(args.batch_fs, -1, 1, dim), means.reshape(args.batch_fs, 1, args.n_ways, dim))
                    soft_sims = torch.softmax(temp_trans * similarities, dim = 2)
                else:
                    similarities = torch.norm(runs[:,:,n_shots:].reshape(args.batch_fs, -1, 1, dim) - means.reshape(args.batch_fs, 1, args.n_ways, dim), dim = 3, p = 2)
                    soft_sims = torch.exp( -1 * temp_trans * similarities)
                for _ in range(n_iter_trans_sinkhorn):
                    soft_sims = soft_sims / soft_sims.sum(dim = 2, keepdim = True) * args.n_ways
                    soft_sims = soft_sims / soft_sims.sum(dim = 1, keepdim = True) * args.n_queries
                new_means = ((runs[:,:,:n_shots].mean(dim = 2) * n_shots + torch.einsum("rsw,rsd->rwd", soft_sims, runs[:,:,n_shots:].reshape(runs.shape[0], -1, runs.shape[3])))) / runs.shape[2]
                if cosine:
                    new_means = new_means / torch.norm(new_means, dim = 2, keepdim = True)
                means = means * alpha_trans + (1 - alpha_trans) * new_means
                if cosine:
                    means = means / torch.norm(means, dim = 2, keepdim = True)
            if cosine:
                winners = torch.max(similarities.reshape(runs.shape[0], runs.shape[1], runs.shape[2] - n_shots, -1), dim = 3)[1]
            else:
                winners = torch.min(similarities.reshape(runs.shape[0], runs.shape[1], runs.shape[2] - n_shots, -1), dim = 3)[1]
            scores += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())
        return utils.stats(scores, "")

def kmeans(args, train_features, features, run_classes, run_indices, n_shots, elements_train=None):
    with torch.no_grad():
        dim = features.shape[2]
        targets = torch.arange(args.n_ways).unsqueeze(1).unsqueeze(0).to(args.device)
        features = utils.preprocess(train_features, features, args.preprocessing, elements_train=elements_train)
        scores = []
        for batch_idx in range(args.n_runs // args.batch_fs):
            runs = generate_runs(features, run_classes, run_indices, batch_idx, args.batch_fs)
            means = torch.mean(runs[:,:,:n_shots], dim = 2)
            for i in range(500):
                similarities = torch.norm(runs[:,:,n_shots:].reshape(args.batch_fs, -1, 1, dim) - means.reshape(args.batch_fs, 1, args.n_ways, dim), dim = 3, p = 2)
                new_allocation = (similarities == torch.min(similarities, dim = 2, keepdim = True)[0]).float()
                new_allocation = new_allocation / new_allocation.sum(dim = 1, keepdim = True)
                allocation = new_allocation
                means = (runs[:,:,:n_shots].mean(dim = 2) * n_shots + torch.einsum("rsw,rsd->rwd", allocation, runs[:,:,n_shots:].reshape(runs.shape[0], -1, runs.shape[3])) * args.n_queries) / runs.shape[2]
            winners = torch.min(similarities.reshape(runs.shape[0], runs.shape[1], runs.shape[2] - n_shots, -1), dim = 3)[1]
            scores += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())
        return utils.stats(scores, "")

def softkmeans(args, train_features, features, run_classes, run_indices, n_shots, elements_train=None):
    with torch.no_grad():
        dim = features.shape[2]
        targets = torch.arange(args.n_ways).unsqueeze(1).unsqueeze(0).to(args.device)
        features = utils.preprocess(train_features, features, args.preprocessing, elements_train=elements_train)
        scores = []
        for batch_idx in range(args.n_runs // args.batch_fs):
            runs = generate_runs(features, run_classes, run_indices, batch_idx, args.batch_fs)
            runs = utils.postprocess(runs, args.postprocessing)
            means = torch.mean(runs[:,:,:n_shots], dim = 2)
            for i in range(30):
                similarities = torch.norm(runs[:,:,n_shots:].reshape(args.batch_fs, -1, 1, dim) - means.reshape(args.batch_fs, 1, args.n_ways, dim), dim = 3, p = 2)
                soft_allocations = F.softmax(-similarities.pow(2)*args.transductive_temperature_softkmeans, dim=2)
                means = torch.sum(runs[:,:,:n_shots], dim = 2) + torch.einsum("rsw,rsd->rwd", soft_allocations, runs[:,:,n_shots:].reshape(runs.shape[0], -1, runs.shape[3]))
                means = means/(n_shots+soft_allocations.sum(dim = 1).reshape(args.batch_fs, -1, 1))
            winners = torch.min(similarities, dim = 2)[1]
            winners = winners.reshape(args.batch_fs, args.n_ways, -1)
            scores += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())
        return utils.stats(scores, "")

def ncm_cosine(args, train_features, features, run_classes, run_indices, n_shots, elements_train=None):
    with torch.no_grad():
        dim = features.shape[2]
        targets = torch.arange(args.n_ways).unsqueeze(1).unsqueeze(0).to(args.device)
        features = utils.preprocess(train_features, features, args.preprocessing, elements_train=elements_train)
        features = utils.sphering(features)
        scores = []
        for batch_idx in range(args.n_runs // args.batch_fs):
            runs = generate_runs(features, run_classes, run_indices, batch_idx, args.batch_fs)
            means = torch.mean(runs[:,:,:n_shots], dim = 2)
            means = utils.sphering(means)
            distances = torch.einsum("bwysd,bwysd->bwys",runs[:,:,n_shots:].reshape(args.batch_fs, args.n_ways, 1, -1, dim), means.reshape(args.batch_fs, 1, args.n_ways, 1, dim))
            winners = torch.max(distances, dim = 2)[1]
            scores += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())
        return utils.stats(scores, "")


# 计算模型提取的特征
def get_features(model, loader, n_aug, device):
    model.eval()
    for augs in range(n_aug):
        all_features, offset, max_offset = [], 1000000, 0
        target_list = []
        for batch_idx, (data, target) in enumerate(loader):
            with torch.no_grad():
                data, target = data.to(device), target.to(device)
                _, features = model(data)
                all_features.append(features)
                offset = min(min(target), offset)
                max_offset = max(max(target), max_offset)
        num_classes = max_offset - offset + 1
        print(".", end='')
        if augs == 0:
            features_total = torch.cat(all_features, dim = 0).reshape(num_classes, -1, all_features[0].shape[1])
        else:
            features_total += torch.cat(all_features, dim = 0).reshape(num_classes, -1, all_features[0].shape[1])
    return features_total / n_aug

# 评估测试
def eval_few_shot(args, train_features, val_features, novel_features, val_run_classes, val_run_indices, novel_run_classes, novel_run_indices, n_shots, transductive = False,elements_train=None):
    if transductive:
        if args.transductive_softkmeans:
            return softkmeans(args, train_features, val_features, val_run_classes, val_run_indices, n_shots, elements_train=elements_train), softkmeans(args, train_features, novel_features, novel_run_classes, novel_run_indices, n_shots, elements_train=elements_train)
        else:
            return kmeans(args, train_features, val_features, val_run_classes, val_run_indices, n_shots, elements_train=elements_train), kmeans(args, train_features, novel_features, novel_run_classes, novel_run_indices, n_shots, elements_train=elements_train)
    else:
        return ncm(args, train_features, val_features, val_run_classes, val_run_indices, n_shots, elements_train=elements_train), ncm(args, train_features, novel_features, novel_run_classes, novel_run_indices, n_shots, elements_train=elements_train)

def update_few_shot_meta_data(args, model, train_clean, novel_loader, val_loader, few_shot_meta_data):
    # 获取特征
    if "M" in args.preprocessing or args.save_features != '':
        train_features = get_features(model, train_clean, args.sample_aug, args.device)
    else:
        train_features = torch.Tensor(0,0,0)
    val_features = get_features(model, val_loader, args.sample_aug, args.device)
    novel_features = get_features(model, novel_loader, args.sample_aug, args.device)

    res = []
    for i in range(len(args.n_shots)):
        res.append(evaluate_shot(args, i, train_features, val_features, novel_features, few_shot_meta_data, model = model))

    return res

def evaluate_shot(args, index, train_features, val_features, novel_features, few_shot_meta_data, model = None, transductive = False):
    (val_acc, val_conf), (novel_acc, novel_conf) = eval_few_shot(args, train_features, val_features,
                                                                 novel_features, few_shot_meta_data["val_run_classes"][index],
                                                                 few_shot_meta_data["val_run_indices"][index],
                                                                 few_shot_meta_data["novel_run_classes"][index],
                                                                 few_shot_meta_data["novel_run_indices"][index],
                                                                 args.n_shots[index], transductive = transductive,
                                                                 elements_train=few_shot_meta_data["elements_train"])
    # 更新最优准确率
    if val_acc > few_shot_meta_data["best_val_acc"][index]:
        if val_acc > few_shot_meta_data["best_val_acc_ever"][index]:
            few_shot_meta_data["best_val_acc_ever"][index] = val_acc

        few_shot_meta_data["best_val_acc"][index] = val_acc
        few_shot_meta_data["best_novel_acc"][index] = novel_acc
    return val_acc, val_conf, novel_acc, novel_conf
