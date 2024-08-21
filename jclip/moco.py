#import torchvision.models as models
#import torch
#import torch.nn as nn
import os
#from torchvision.models import resnet50
import jittor as jt
from jittor.models.resnet import  *
import jittor.nn as nn
#import torchvision.models as models

# 创建 ResNet50 模型实例
#resnet50_pytorch = models.resnet50(pretrained=True)
'''def load_moco(pretrain_path):
    print("=> creating model")
    model = Resnet50() # jittor.models.Resnet50(pretrained=False, **kwargs)
    # 参数：pretrained (bool): 表示是否加载预训练的ResNet50模型。默认为 False。如果设为 True, 函数会自动下载并加载预训练的ResNet50模型。**kwargs: 可变参数, 允许用户传递额外的、自定义的参数给 _resnet 函数。
    # 返回值：返回一个ResNet50模型实例。如果 pretrained=True, 则返回的模型将加载预训练权重；否则, 返回一个未经训练的ResNet50模型。
    linear_keyword = 'fc'
    if os.path.isfile(pretrain_path):
        print("=> loading checkpoint '{}'".format(pretrain_path))
        checkpoint = jt.load(pretrain_path)   ## jittor.load(path: str)  从指定路径 path 加载一个模型。参数:path (str) :需要加载模型的具体路径，此参数为必需。返回值:返回加载后的模型。
        state_dict = checkpoint["state_dict"]
        for k in list(state_dict.keys()):
            if k.startswith('backbone') and not k.startswith('backbone.%s' % linear_keyword):
                # remove prefix
                state_dict[k[len("backbone."):]] = state_dict[k]
            # delete renamed or unused k
            elif k.startswith('momentum_encoder') and not k.startswith('momentum_encoder.%s' % linear_keyword):
                state_dict[k[len('momentum_encoder.'):]] = state_dict[k]
            del state_dict[k]

        msg = model.load_state_dict(state_dict)
        print(msg)
        assert set(msg.missing_keys) == {f"{linear_keyword}.weight", f"{linear_keyword}.bias"}

        print("=> loaded pre-trained model '{}'".format(pretrain_path))
    else:
        print("=> no checkpoint found at '{}'".format(pretrain_path))
        raise FileNotFoundError
    model.fc = nn.Identity() # 该类用于占位，即它会输出与输入相同的张量。这个模块不会对数据进行任何改变或计算。
    return model, 2048'''



### 目前 改成jittor后的代码
def load_moco(pretrain_path):
    print("=> creating model")
    model = resnet50()
    linear_keyword = 'fc'

    # if "steps" in model.layer1

    if os.path.isfile(pretrain_path):
        print("=> loading checkpoint '{}'".format(pretrain_path))
        state_dict = jt.load(pretrain_path)
        #print(list(state_dict.keys()))
        # print(state_dict)
        # print(-1)
        #state_dict = checkpoint["state_dict"]

        new_state_dict = {}
        for k in list(state_dict.keys()):
            
            if k.startswith('backbone') and not k.startswith(f'backbone.{linear_keyword}'):
                # print(k)
                new_state_dict[k[len("backbone."):]] = state_dict[k]
            elif k.startswith('momentum_encoder') and not k.startswith(f'momentum_encoder.{linear_keyword}'):
                new_state_dict[k[len('momentum_encoder.'):]] = state_dict[k]
        #print(-1)
        #print(model)
        #print(list(new_state_dict.keys()))
        model.load_parameters(new_state_dict)
        
        # print('88888888888888')
        print("=> loaded pre-trained model '{}'".format(pretrain_path))
    else:
        print("=> no checkpoint found at '{}'".format(pretrain_path))
        raise FileNotFoundError
    
    model.fc = nn.Identity()
    return model, 2048
# import torchvision.models as models
# import torch
# import torch.nn as nn
# import os
# from torchvision.models import resnet50
# def load_moco(pretrain_path):
#     print("=> creating model")
#     model = resnet50()
#     linear_keyword = 'fc'
#     if os.path.isfile(pretrain_path):
#         print("=> loading checkpoint '{}'".format(pretrain_path))
#         checkpoint = torch.load(pretrain_path, map_location="cpu")
#         state_dict = checkpoint["state_dict"]
#         model.load_state_dict(state_dict)
#         for k in list(state_dict.keys()):
#             if k.startswith('backbone') and not k.startswith('backbone.%s' % linear_keyword):
#                 # remove prefix
#                 state_dict[k[len("backbone."):]] = state_dict[k]
#             # delete renamed or unused k
#             elif k.startswith('momentum_encoder') and not k.startswith('momentum_encoder.%s' % linear_keyword):
#                 state_dict[k[len('momentum_encoder.'):]] = state_dict[k]
#             del state_dict[k]

#         msg = model.load_state_dict(state_dict, strict=False)
#         print(msg)
#         assert set(msg.missing_keys) == {f"{linear_keyword}.weight", f"{linear_keyword}.bias"}

#         print("=> loaded pre-trained model '{}'".format(pretrain_path))
#     else:
#         print("=> no checkpoint found at '{}'".format(pretrain_path))
#         raise FileNotFoundError
#     model.fc = nn.Identity()
#     return model, 2048



#  pytorch 正确版
'''def load_moco(pretrain_path):
    print("=> creating model")
    model = resnet50()
    linear_keyword = 'fc'
    if os.path.isfile(pretrain_path):
        print("=> loading checkpoint '{}'".format(pretrain_path))
        checkpoint = jt.load(pretrain_path)
        state_dict = checkpoint
        for k in list(state_dict.keys()):
            if k.startswith('backbone') and not k.startswith('backbone.%s' % linear_keyword):
                # remove prefix
                state_dict[k[len("backbone."):]] = state_dict[k]
            # delete renamed or unused k
            elif k.startswith('momentum_encoder') and not k.startswith('momentum_encoder.%s' % linear_keyword):
                state_dict[k[len('momentum_encoder.'):]] = state_dict[k]
            del state_dict[k]

        msg = model.load_state_dict(state_dict)
        print(msg)
        assert set(msg.missing_keys) == {f"{linear_keyword}.weight", f"{linear_keyword}.bias"}

        print("=> loaded pre-trained model '{}'".format(pretrain_path))
    else:
        print("=> no checkpoint found at '{}'".format(pretrain_path))
        raise FileNotFoundError
    model.fc = nn.Identity()
    return model, 2048'''










if __name__ == "__main__":
    
    model = load_moco("resnet50", ).cuda() 
    #print(model)
    print(model(jt.random((32,3,224,224)).cuda()).shape)

    
    
    