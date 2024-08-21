import os
import shutil
import random

import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

model.eval()  # 设置模型为评估模式


# 从train.txt读取文件地址
def get_all(file_path="Dataset/train.txt"):
    file_train = open(file_path, 'r').read().splitlines()
    all_data = {}
    for line in file_train:
        path = line.split(' ')[0]
        index = line.split(' ')[1]
        name = path.split('/')[2]
        if name not in all_data:
            all_data[name] = [(path, index)]
        else:
            all_data[name].append((path, index))
    return all_data


# 去重操作
def remove_train(data_all, data_remove={}):
    ans = {}
    for key in data_all:
        if key in data_remove:
            list1 = data_all[key]
            list2 = data_remove[key]
            new_list = [tup for tup in list1 if tup not in list2]
            if new_list:
                ans[key] = new_list

            # 每个类随机选择Num个元素
    return ans


def select(data, num):
    ans = {}
    for key in data:
        if len(data[key]) < num:
            print(key)
        ans[key] = random.sample(data[key], num)
    return ans


# 示例数据
# vectors = torch.randn(10, 3)  # 假设有10个3维向量
def find_k_nearest(vectors, k):
    # 1. 计算中心向量（均值向量）
    print(vectors.shape)
    center = vectors.mean(dim=0)

    # 2. 计算每个向量到中心向量的距离（欧氏距离）
    # 使用torch.cdist函数可以更方便地计算距离矩阵，但这里为了展示过程，我们使用显式的方法
    distances = torch.sqrt(((vectors - center) ** 2).sum(dim=1))

    # 3. 对距离进行排序，并获取排序后的索引
    _, indices = torch.sort(distances)

    # 4. 选取距离最小的k个向量的索引
    k_nearest_indices = indices[:2].numpy().tolist()
    k_nearest_indices.append(indices[-1])
    k_nearest_indices.append(indices[-2])
    k_nearest_indices = torch.tensor(k_nearest_indices)
    return k_nearest_indices


# 对图片经过CLIP编码
def encoder_image(img_path):
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    image_feature = model.encode_image(image)
    return image_feature


def encoder_select(data, num):
    ans = {}
    for key in data:  # 遍历每个种类
        vectors = []
        for file in data[key]:  # 将该类图片进行编码
            vectors.append(encoder_image(file[0]))
        # 目前可能会出错
        vectors = torch.cat(vectors, dim=0)
        indexes = find_k_nearest(vectors, num)
        ans[key] = [data[key][i] for i in indexes]
        del indexes
        del vectors
    return ans


# 确保目标路径存在
def make_sure(dir_path):
    # 检查目录是否存在
    if os.path.exists(dir_path):
        # 如果目录存在，检查是否为空
        if os.listdir(dir_path):
            # 如果目录不为空，删除目录及其内容
            shutil.rmtree(dir_path)
            print(f"目录 {dir_path} 已删除")
        else:
            # 如果目录为空，什么都不做（或者你可以打印一条消息）
            print(f"目录 {dir_path} 存在但为空")
    else:
        # 如果目录不存在，创建目录
        os.makedirs(dir_path)
        print(f"目录 {dir_path} 已创建")


def make_dic(f):
    # 检查目录是否存在
    if not os.path.exists(f):
        # 如果目录不存在，创建目录
        os.makedirs(f)
        print(f"目录 {f} 已创建")


def copy_image(data_all, tar, rec):
    for key in data_all:
        for v in data_all[key]:
            file = v[0]
            index = v[1]
            p = file.split('/')
            make_dic(os.path.join(tar))
            name = p[1] + "_" + p[2] + "_" + p[3]
            t = os.path.join(tar, name)
            shutil.copy2(file, t)
            rec.write(name + " " + index + "\n")


def copy_image_fold(data_all, tar, rec):
    for key in data_all:
        for v in data_all[key]:
            file = v[0]
            index = v[1]
            p = file.split('/')
            # make_dic(os.path.join(tar, p[1]))
            # make_dic(os.path.join(tar, p[1], p[2]))
            make_dic(os.path.join(tar, p[2]))
            t = os.path.join(tar, p[2], p[3])
            shutil.copy2(file, t)
            rec.write(t + " " + index + "\n")


def copy_image_fold2(data_all, tar, rec, rec2):
    for key in data_all:
        for v in data_all[key]:
            file = v[0]
            index = v[1]
            p = file.split('/')
            # make_dic(os.path.join(tar, p[1]))
            # make_dic(os.path.join(tar, p[1], p[2]))
            make_dic(os.path.join(tar, p[2]))
            t = os.path.join(tar, p[2], p[3])
            shutil.copy2(file, t)
            rec.write(t + " " + index + "\n")
            rec2.write(file + "\n")


def save_change(gen, t, r, num, data_all):
    d1 = select(data_all, num)
    t = os.path.join(gen, t)
    make_sure(t)
    rr = open(r, 'w')
    copy_image_fold(d1, t, rr)
    return remove_train(data_all, d1)


def save_train(gen, t, r, r2, num, data_all):
    d1 = encoder_select(data_all, num)
    t = os.path.join(gen, t)
    make_sure(t)
    rr = open(r, 'w')
    rp = open(r2, 'w')
    copy_image_fold2(d1, t, rr, rp)
    return remove_train(data_all, d1)


if __name__ == '__main__':
    train_file = "Dataset/train.txt"  # train.txt的路径
    root = "DataSets"
    make_sure(root)
    target = "Valid"
    record = "Valid.txt"
    train_data = get_all(train_file)
    # for i in train_data.keys():
    #     print(i,len(train_data[i]))
    # 从候选列表里删除实际训练的数据

    # remove_file = "实际训练的数据列表"
    # remove_d = get_all(remove_file)
    # train_data = remove_train(train_data, remove_d)
    target1 = 'Train'
    record1 = 'Train.txt'
    record_path = 'Train_path.txt'
    target2 = 'Test'
    record2 = 'Test.txt'
    # 每个类选取多少张
    number = 8
    t1 = save_train(root, target1, record1, record_path, 4, train_data)
    t2 = save_change(root, target, record, number, t1)
    t3 = save_change(root, target2, record2, number, t2)
