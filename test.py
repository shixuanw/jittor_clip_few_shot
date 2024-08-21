from jclip import amu
import jittor
import jittor as jt
from jittor.transform import Compose, Resize, CenterCrop, ToTensor
import os
import tqdm
from DataSets.testa import TestA
from utils import *
from jclip.moco import load_moco
from jclip.amu import *
from PIL import Image
from parse_args import parse_args
import jclip
import random
def _convert_image_to_rgb(image):
    return image.convert("RGB")
parser = parse_args()
args = parser.parse_args()
cache_dir = os.path.join('./caches', args.dataset)
os.makedirs(cache_dir, exist_ok=True)
args.cache_dir = cache_dir



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

imagenet = TestA(args.shots)
train_loader_feature = jittor.dataset.DataLoader(imagenet.train, batch_size=16, shuffle=False)
jclip_weights = gpt_clip_classifier(imagenet.classnames, jclip_model, imagenet.template)
aux_features, aux_labels = load_aux_weight(args, aux_model, train_loader_feature, tfm_norm=tfm_aux)


model = amu.AMU_Model(
        clip_model=jclip_model,
        aux_model=aux_model,
        sample_features=[aux_features, aux_labels],
        clip_weights=jclip_weights,
        feat_dim=args.feat_dim,
        class_num=args.num_classes,
        lambda_merge=args.lambda_merge,
        alpha=args.alpha,
        uncent_type=args.uncent_type,
        uncent_power=args.uncent_power
    )
total_params = sum(p.numel() for p in model.parameters())
print(f"模型的总参数量: {total_params/(1024*1024)}M")
model.cuda()
model.requires_grad_(False)
model.load_state_dict(jt.load('final_adapter_4shots.pkl'))

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