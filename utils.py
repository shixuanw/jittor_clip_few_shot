import logging
import datetime
from PIL import Image
from tqdm import tqdm

import jclip
import jittor
from jittor.transform import Compose, Resize, CenterCrop, ToTensor, RandomResizedCrop, RandomHorizontalFlip

BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


tfm_train_base = Compose([
    RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=BICUBIC),
    RandomHorizontalFlip(p=0.5),
    ToTensor()
]
)

tfm_test_base = Compose([
    Resize(224),
    CenterCrop(224),
    _convert_image_to_rgb,
    ToTensor(),
])

# def cls_acc(output, target, topk=1):
#     pred = output.topk(topk, 1, True, True)[1].t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))
#     acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
#     acc = 100 * acc / target.shape[0]
#     return acc


# def cls_acc(output, target, topk=1):
#     pred = output.topk(topk, 1, True, True)[1].t()
#     # 将target扩展为与pred相同的形状以便比较
#     # target_expanded = target.view(1,-1).expand_as(pred)
#     print(pred.shape)
#     # print(target_expanded)2
#     # 比较预测索引和真实标签
#     correct = jittor.equal(pred, target.view(1,-1).expand_as(pred))
#     # print(correct)
#     # 计算准确率，注意Jittor中不需要.cpu().numpy()，因为我们可以直接在Jittor张量上进行操作
#     num_correct = jittor.sum(correct[: topk].reshape(-1).float(),dim=0,keepdims=True).item()  # .item()将标量张量转换为Python数值
#     # acc = 100.0 * num_correct / target.numel()  # 使用numel()获取target中的元素总数
#     acc = 100.0 * num_correct / target.shape[0]
#     return acc
import torch


def equal_num(x, y):
    # x = x.reshape(1, -1)
    # y = y.reshape(1, -1)
    num = 0
    # print(-1)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i, j] == y[i, j]:
                num += 1
    # print(num)
    return num


import jittor as jt


def cls_acc(output, target, topk=1):
    with jittor.no_grad():
        maxk = max(topk, 1)
        batch_size = target.shape[0]

        # 获取前K个预测结果
        _, pred = jt.topk(output, maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = (pred == target.view(1, -1).expand_as(pred))

        # 计算Top-K准确率
        correct_k = correct[:maxk].reshape(-1).float().sum(0)
        accuracy = correct_k * 100.0 / batch_size
        temp = accuracy.item()
        return temp


def gpt_clip_classifier(classnames, clip_model, template):
    with jittor.no_grad():
        clip_weights = []
        for classname in classnames:
            # Tokenize the prompts
            #classname = classname.replace('_', ' ')
            #texts = [t.format(classname) for t in template]
            texts = jclip.tokenize(classname)
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = jittor.stack(clip_weights, dim=1)
    return clip_weights


def build_cache_model(clip_model, train_loader_cache, tfm_norm):
    cache_keys = []
    cache_values = []

    with jittor.no_grad():
        # Data augmentation for the cache model
        # 由20改成了1
        for augment_idx in range(1):
            train_features = []

            print('Augment Epoch: {:} / {:}'.format(augment_idx, 20))
            for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                # images = images.cuda()
                image_features = clip_model.encode_image(tfm_norm(images))
                train_features.append(image_features)
                if augment_idx == 0:
                    # target = target.cuda()
                    cache_values.append(target)
            cache_keys.append(jittor.cat(train_features, dim=0).unsqueeze(0))

    cache_keys = jittor.cat(cache_keys, dim=0).mean(dim=0)
    cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
    cache_keys = cache_keys.permute(1, 0)
    cache_values = jittor.nn.one_hot(jittor.cat(cache_values, dim=0), 403).half()
    cache_values = jittor.float32(cache_values)
    # torch.save(cache_keys, 'D:\\深度学习\\AMU_merge\\caches\\ImageNet' + '\\keys_' + str(4) + "shots.pt")
    # torch.save(cache_values, 'D:\\深度学习\\AMU_merge\\caches\\ImageNet' + '\\values_' + str(4) + "shots.pt")

    return cache_keys, cache_values


def load_aux_weight(args, model, train_loader_cache, tfm_norm):
    if args.load_aux_weight == False:
        aux_features = []
        aux_labels = []
        with jittor.no_grad():
            for augment_idx in range(args.augment_epoch):
                aux_features_current = []
                print('Augment Epoch: {:} / {:}'.format(augment_idx, args.augment_epoch))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    # images = images.cuda()
                    image_features = model(tfm_norm(images))
                    aux_features_current.append(image_features)
                    if augment_idx == 0:
                        # target = target.cuda()
                        aux_labels.append(target)
                aux_features.append(jittor.cat(aux_features_current, dim=0).unsqueeze(0))

        aux_features = jittor.cat(aux_features, dim=0).mean(dim=0)
        aux_features /= aux_features.norm(dim=-1, keepdim=True)

        aux_labels = jittor.cat(aux_labels)

        jittor.save(aux_features, args.cache_dir + f'/aux_feature_' + str(args.shots) + "shots.pkl")
        jittor.save(aux_labels, args.cache_dir + f'/aux_labels_' + str(args.shots) + "shots.pkl")

    else:
        aux_features = jittor.load(args.cache_dir + f'/aux_feature_' + str(args.shots) + "shots.pkl")
        aux_labels = jittor.load(args.cache_dir + f'/aux_labels_' + str(args.shots) + "shots.pkl")
    return aux_features, aux_labels


def load_test_features(args, split, model, loader, tfm_norm, model_name):
    if args.load_pre_feat == False:
        features, labels = [], []
        with jittor.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                # images, target = images.cuda(), target.cuda()
                if hasattr(model, 'encode_image') and callable(getattr(model, 'encode_image')):
                    image_features = model.encode_image(tfm_norm(images))  # for clip model
                else:
                    image_features = model(tfm_norm(images))
                features.append(image_features)
                labels.append(target)

        features, labels = jittor.cat(features), jittor.cat(labels)
        # features = features.cuda()
        jittor.save(features, args.cache_dir + f"/{model_name}_" + split + "_f.pt")
        jittor.save(labels, args.cache_dir + f"/{model_name}_" + split + "_l.pt")

    else:
        features = jittor.load(args.cache_dir + f"/{model_name}_" + split + "_f.pt")
        labels = jittor.load(args.cache_dir + f"/{model_name}_" + split + "_l.pt")
    return features, labels


def config_logging(args):
    logger = logging.getLogger()  # root logger
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M')
    now = datetime.datetime.now().strftime("%m-%d-%H_%M")
    # FileHandler
    fh = logging.FileHandler(f'result/{args.exp_name}_{now}.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # StreamHandler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def search_hp(cfg, cache_keys, cache_values, clip_test_features, aux_test_features, labels, clip_weights, model):
    beta_list = [i * (12 - 0.1) / 200 + 0.01 for i in range(200)]
    alpha_list = [i * (100 - 0.1) / 2000 + 0.01 for i in range(2000)]

    best_acc = 0.0
    best_beta, best_alpha = 0, 0
    for beta in beta_list:
        for alpha in alpha_list:

            affinity = clip_test_features @ cache_keys
            #  2992,512   512，4488   ->  2992,4488
            return_dict = model(
                clip_features=clip_test_features,
                aux_features=aux_test_features,
                labels=labels
            )
            # print(f"return_dict size:{jittor.size(return_dict['logits'])}")
            # cache_keys:[512,4488,] cache_values:[4488,374,] jiclip_test_features:[2992,512,]
            # aux_test_features:[4488,2048,] test_labels:[2992,]
            # print(f"type cache_values:{cache_values.dtype}")
            cache_values = jittor.float32(cache_values)
            # print(f"type cache_values:{cache_values.dtype}")
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            #  2992,4488  *4488,403   ->2992,403
            clip_logits = return_dict['logits']
            # print(f"cache_logits:{jittor.size(cache_logits)} clip_logits:{jittor.size(clip_logits)}")
            tip_logits = clip_logits + cache_logits * alpha
            acc = cls_acc(tip_logits, labels)

            if acc > best_acc:
                print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                best_acc = acc
                best_beta = beta
                best_alpha = alpha

    print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha
