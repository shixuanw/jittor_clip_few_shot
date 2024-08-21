import os
import random
from tqdm import tqdm

import jittor
import jittor.transform as transform
# from datasets.imagenet import ImageNet
from DataSets.testa import TestA
import jclip
from utils import *
from jclip.moco import load_moco
from jclip.amu import *
from parse_args import parse_args


# jt.flags.use_cuda=1

def freeze_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def train_one_epoch(model, data_loader, optimizer, scheduler, logger):
    # Train
    model.train()
    model.apply(freeze_bn)  # freeze BN-layer
    correct_samples, all_samples = 0, 0
    loss_list = []
    loss_aux_list = []
    loss_merge_list = []

    # origin image
    for i, (images, target) in enumerate(tqdm(data_loader)):
        # images, target = images.cuda(), target.cuda()
        # print(type(images))
        # print(images.shape)
        # print(images.shape)
        return_dict = model(images, labels=target)
        # print(return_dict['logits'].shape)
        acc = cls_acc(return_dict['logits'], target)
        correct_samples += acc / 100 * len(return_dict['logits'])
        all_samples += len(return_dict['logits'])

        loss_list.append(return_dict['loss'].item())
        loss_aux_list.append(return_dict['loss_aux'].item())
        loss_merge_list.append(return_dict['loss_merge'].item())

        optimizer.zero_grad()
        # return_dict['loss'].backward()   super_cui改
        optimizer.backward(return_dict['loss'])
        optimizer.step()
        scheduler.step()
        # jt.sync_all()
        # jt.display_memory_info()
        # jt.gc()

    # current_lr = scheduler.get_last_lr()[0] jittor没有从scheduler获取学习率的函数
    current_lr = optimizer.lr
    logger.info('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples,
                                                                         correct_samples, all_samples,
                                                                         sum(loss_list) / len(loss_list)))
    logger.info("""Loss_aux: {:.4f}, Loss_merge: {:.4f}""".format(sum(loss_aux_list) / len(loss_aux_list),
                                                                  sum(loss_merge_list) / len(loss_merge_list)))
    # jt.sync_all()
    # jt.display_memory_info()
    # jt.gc()


def _convert_image_to_rgb(image):
    return image.convert("RGB")


import psutil
import gc


def train_and_eval(args, logger, model, jclip_test_features, aux_test_features, test_labels, train_loader_F):
    model.cuda()
    model.requires_grad_(False)
    model.aux_adapter.requires_grad_(True)

    optimizer = jittor.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    scheduler = jittor.lr_scheduler.CosineAnnealingLR(optimizer, args.train_epoch * len(train_loader_F))

    best_acc = 0.0
    total_cpu_ram = psutil.virtual_memory().total
    print(f"Total CPU RAM: {total_cpu_ram / (1024 ** 3):.2f} GB")
    for train_idx in range(15):
        logger.info('Train Epoch: {:} / {:}'.format(train_idx, args.train_epoch))
        train_one_epoch(model, train_loader_F, optimizer, scheduler, logger)
        total_cpu_ram = psutil.virtual_memory().total
        print(f"Total CPU RAM: {total_cpu_ram / (1024 ** 3):.2f} GB")
        gc.collect()
        #jt.sync_all()
        #jt.display_memory_info()
        # gc.collect()
        # Eval
        model.eval()
        with jittor.no_grad():
            # print(-1)
            return_dict = model(
                # jclip_features=jclip_test_features, 多加了一个j
                clip_features=jclip_test_features,
                aux_features=aux_test_features,
                labels=test_labels
            )
            # print(return_dict['logits'].shape)
            # print(test_labels.shape)
            #print("aosdjaifgjaig")
            acc = cls_acc(return_dict['logits'], test_labels)
            #print("oooooooooook")
            acc_aux = cls_acc(return_dict['aux_logits'], test_labels)
        logger.info("----- Aux branch's Test Acc: {:.2f} ----".format(acc_aux))
        logger.info("----- AMU's Test Acc: {:.2f} -----\n".format(acc))

        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            jittor.save(model.state_dict(), args.cache_dir + f"/best_adapter_" + str(args.shots) + "shots.pkl")
            jittor.save(model.state_dict(), '/content/drive/MyDrive/jittor_model' + f"/best_adapter_" + str(args.shots) + "shots.pkl")
    jittor.save(model.state_dict(), args.cache_dir + f"/final_adapter_" + str(args.shots) + "shots.pkl")
    jittor.save(model.state_dict(), '/content/drive/MyDrive/jittor_model' + f"/final_adapter_" + str(args.shots) + "shots.pkl")
    logger.info(f"----- Best Test Acc: {best_acc:.2f}, at epoch: {best_epoch}.-----\n")
    # ------------------------#

    model.load_state_dict(jittor.load(args.cache_dir + f"/final_adapter_" + str(args.shots) + "shots.pkl"))
    #print(
        #f"cache_keys:{jittor.size(cache_keys)} cache_values:{jittor.size(cache_values)} jiclip_test_features:{jittor.size(jclip_test_features)} aux_test_features:{jittor.size(aux_features)} test_labels:{jittor.size(test_labels)}")
    # beta, alpha = search_hp(args, cache_keys, cache_values, jclip_test_features,aux_test_features, test_labels, jclip_weights,model)
    #beta = 2.81
    #alpha = 0.91
    classes = open('classes_b.txt').read().splitlines()
    # remove the prefix Animal, Thu-dog, Caltech-101, Food-101
    new_classes = []
    for c in classes:
        c = c.split(' ')[0]
        if c.startswith('Animal'):
            c = c[7:]
        if c.startswith('Thu-dog'):
            c = c[8:]
        if c.startswith('Caltech-101'):
            c = c[12:]
        if c.startswith('Food-101'):
            c = c[9:]
        new_classes.append(c)
    train_preprocess = Compose([
        Resize(224),
        CenterCrop(224),
        _convert_image_to_rgb,
        ToTensor(),
    ])
    split = 'TestSetB'
    imgs = os.listdir(split)
    save_file = open('result.txt', 'w')
    for img in tqdm(imgs):
        img_path = os.path.join(split, img)
        image = Image.open(img_path)
        image = train_preprocess(image)
        image = jittor.reshape(image, ((1, -1, 224, 224)))
        # print(image.shape)
        # with jittor.no_grad():
        #     image_features = jclip_model.encode_image(image)
        #     image_features /= image_features.norm(dim=-1, keepdim=True)
        # affinity = image_features @ cache_keys
        # cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        return_dict = model(image)
        jclip_logits = return_dict["logits"]
        tip_logits = jclip_logits #+ cache_logits * alpha
        pred = tip_logits.topk(5, 1, True, True)
        ans = []
        for i in pred[1]:
            vp = 0
            for j in i:
                v = imagenet.tmp[j]  # 字符串名字
                tp = 0
                ok = 0
                for k in range(len(new_classes)):
                    if new_classes[k] == v:
                        tp = k
                        ok = 1
                        break
                ans.append(tp)
                vp += 1
                if (ok == 0):
                    print(ok)
                    while 1:
                        print("No")
        save_file.write(img + ' ' + ' '.join([str(p) for p in ans]) + '\n')


if __name__ == '__main__':
    # Load config file
    parser = parse_args()
    args = parser.parse_args()

    cache_dir = os.path.join('./caches', args.dataset)
    os.makedirs(cache_dir, exist_ok=True)
    args.cache_dir = cache_dir

    logger = config_logging(args)
    logger.info("\nRunning configs.")
    args_dict = vars(args)
    message = '\n'.join([f'{k:<20}: {v}' for k, v in args_dict.items()])
    logger.info(message)
    # jclip
    jclip_model, preprocess = jclip.load("ViT-B/32")
    # jclip_model=jittor.load("C:/Users/quark404/.cache/clip/ViT-B-32.pkl")
    jclip_model.cuda()
    jclip_model.eval()
    # AUX MODEL
    import numpy as np

    aux_model, args.feat_dim = load_moco("mocov3.pkl")  # Aux model path
    # aux_model, args.feat_dim = load_moco("100/archive/data.pkl")#Aux model path
    aux_model.cuda()
    aux_model.eval()

    # ImageNet dataset
    random.seed(args.rand_seed)

    logger.info("Loading ImageNet dataset....")
    print(args.shots)
    imagenet = TestA(20)
    test_loader = jittor.dataset.DataLoader(imagenet.test, batch_size=16, shuffle=False)
    train_loader_cache = jittor.dataset.DataLoader(imagenet.train, batch_size=256, shuffle=False)
    train_loader = jittor.dataset.DataLoader(imagenet.train, batch_size=8, shuffle=True)
    train_loader_feature = jittor.dataset.DataLoader(imagenet.train, batch_size=16, shuffle=False)
    #cache_keys, cache_values = build_cache_model(jclip_model, train_loader_cache, tfm_norm=tfm_clip)
    # Textual features
    logger.info("Getting textual features as jclip's classifier...")
    jclip_weights = gpt_clip_classifier(imagenet.classnames, jclip_model, imagenet.template)

    # Load visual features of few-shot training set
    logger.info("Load visual features of few-shot training set...")
    aux_features, aux_labels = load_aux_weight(args, aux_model, train_loader_feature, tfm_norm=tfm_aux)
    logger.info(f"aux_features size:{jt.size(aux_features)},aux_labels size:{jt.size(aux_labels)}")
    # Pre-load test features
    logger.info("Loading visual features and labels from test set.")

    logger.info("Loading jclip test feature.")
    test_jclip_features, test_labels = load_test_features(args, "test", jclip_model, test_loader, tfm_norm=tfm_clip,
                                                          model_name='jclip')
    logger.info(f"Loading AUX test feature.")
    test_aux_features, test_labels = load_test_features(args, "test", aux_model, test_loader, tfm_norm=tfm_aux,
                                                        model_name='aux')

    test_jclip_features = test_jclip_features
    test_aux_features = test_aux_features

    # zero shot
    tmp = test_jclip_features / test_jclip_features.norm(dim=-1, keepdim=True)
    l = 100. * tmp @ jclip_weights
    # print(f"{l.argmax(dim=-1).eq(test_labels).sum().item()}/ {len(test_labels)} = {l.argmax(dim=-1).eq(test_labels).sum().item()/len(test_labels) * 100:.2f}%")

    # build amu-model
    print(f'------------------------feat_dim={args.feat_dim}')
    model = AMU_Model(
        clip_model=jclip_model,
        aux_model=aux_model,
        sample_features=[aux_features, aux_labels],
        clip_weights=jclip_weights,
        feat_dim=args.feat_dim,
        class_num=403,
        lambda_merge=args.lambda_merge,
        alpha=args.alpha,
        uncent_type=args.uncent_type,
        uncent_power=args.uncent_power
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型的总参数量: {total_params/(1024*1024)}M")
    train_and_eval(args, logger, model, test_jclip_features, test_aux_features, test_labels, train_loader)