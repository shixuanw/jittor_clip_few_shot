# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.transforms import Compose, Normalize
import jittor.transform as transform
import jittor as jt
import jittor.nn as nn


# def logit_normalize(logit):
#     # logits_std = torch.std(logit, dim=1, keepdim=True)
#     logits_std = jt.std(logit)  # jittor中没有选定维度的参数，不知道这里会不会出现问题
#     # logits_mean = torch.mean(logit, dim=1, keepdim=True)
#     logits_mean = logit.mean(dim=1, keepdims=True)
#     logit = (logit - logits_mean) / logits_std
#     return logit
def std(tensor, dim=None, keepdim=False, unbiased=True):
    # Calculate mean
    mean = jt.mean(tensor, dim=dim, keepdims=True)
    # Calculate variance
    variance = jt.mean((tensor - mean) ** 2, dim=dim, keepdims=keepdim)

    # If unbiased is True, we need to adjust for Bessel's correction
    if unbiased and dim is not None:
        num_elements = tensor.shape[dim]
        variance = variance * num_elements / (num_elements - 1)

    # Return standard deviation
    return jt.sqrt(variance)


def logit_normalize(logit):
    logits_std = std(logit, dim=1, keepdim=True)
    logits_mean = jt.mean(logit, dim=1, keepdim=True)
    logit = (logit - logits_mean) / logits_std
    return logit

def uncertainty(logits, type, power):
    # softmax_fun = nn.softmax(dim=-1) # sofemax-norm to get probability distribution
    # logits = softmax_fun(logits)
    logits = nn.softmax(logits, dim=-1)
    if type == 'entropy':
        # entropy = -jt.sum(logits * logits.log2(), dim=-1, keepdims=True) / torch.log2(torch.tensor(logits.shape[-1]).float())
        entropy = -jt.sum(logits * logits.log2(), dim=-1, keepdims=True) / jt.log2(jt.float32(logits.shape[-1]))
        entropy = (entropy * power).exp()
        return entropy
    elif type == 'energy':
        max_values = logits.max(dim=-1, keepdims=True)
        logits = logits - max_values
        tau = 2
        # energy = tau * (torch.log(torch.sum(torch.exp(logits / tau), dim=-1, keepdim=True)) + max_values)
        energy = tau * (jt.log(jt.sum(jt.exp(logits / tau), dim=-1, keepdims=True)) + max_values)
        return 1.0 / (energy ** power)
    elif type == 'max':
        max_values = logits.max(dim=-1, keepdims=True)
        return 1.0 / (max_values) ** power
    elif type == 'max-min':
        diff = logits.max(dim=-1, keepdims=True) - logits.min(dim=-1, keepdims=True)
        return 1.0 / diff ** power
    elif type == 'var':
        # variance = torch.std(logits, dim=-1, keepdim=True)
        variance = jt.std(logits)
        return variance
    elif type == 'top5':
        top2 = logits.topk(5, dim=-1).values
        confidence = (top2[:, 0] - top2[:, -1]).unsqueeze(-1)
        return 1.0 / (confidence) ** power

    elif type == 'moment':
        # mu = torch.mean(logits, dim=-1, keepdim=True)
        mu = jt.mean(logits, dim=-1, keepdims=True)
        # sigma = torch.std(logits, dim=-1, keepdim=True)
        sigma = jt.std(logits)
        normalized_logits = (logits - mu) / sigma
        # moment_4 = torch.mean(normalized_logits ** 4, dim=-1, keepdim=True)
        moment_4 = jt.mean(normalized_logits ** 4, dim=-1, keepdims=True)
        # return 1 / ((moment_4 / 250) ** power)
        return 1 / ((moment_4.divide(jt.var(moment_4.shape).fill(250))) ** power)

        # return 1.5 - 0.12 * moment_4
        # return filp(moment_4)
        # return (- moment_4 * power).exp()
    elif type == 'none':
        return jt.float32(1.0)
    else:
        raise RuntimeError('Invalid uncertainty type.')


class Linear_Adapter(nn.Module):
    def __init__(self, feat_dim, class_num, sample_features=None):
        super().__init__()
        self.fc = nn.Linear(feat_dim, class_num, bias=False)
        # init
        if sample_features is not None:
            print('init adapter weight by training samples...')
            aux_features, aux_labels = sample_features[0], sample_features[1]
            aux_features = aux_features

            # init_weight = torch.zeros(feat_dim, class_num, device=aux_features.device)
            init_weight = jt.zeros((feat_dim, class_num))
            #print(f"-----------init_weight size:{jt.size(init_weight)};aux_labels size:{jt.size(aux_labels)};aux_fuatures size:{jt.size(aux_features)}")
            #print(f"aux_fuatures[0] size{jt.size(aux_features[0])}")
            #print(f"--------init_weight[:,aux_labels[0]] size:{jt.size(init_weight[:, aux_labels[0]])}")
            for i in range(len(aux_labels)):
                # print(f"--------init_weight[:,aux_labels[0]] size:{jt.size(init_weight[:,aux_labels[i]])}")
                init_weight[:, aux_labels[i]] += jt.unsqueeze(aux_features[i], 1)

            feat_per_class = len(aux_labels) / class_num
            init_weight = init_weight / feat_per_class
            self.fc.weight = nn.Parameter(init_weight.t())
        else:
            print('init adapter weight by random...')

    def execute(self, feat):
        return self.fc(feat)


# tfm_clip = transform.Compose([transform.image_normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])
# tfm_aux = transform.Compose([transform.image_normalize(,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
# tfm_clip = transform.Compose([transform.image_normalize(,mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

tfm_clip = transform.Compose([
    transform.ImageNormalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
])

# tfm_aux 对应的 Jittor 代码
tfm_aux = transform.Compose([
    transform.ImageNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class AMU_Model(nn.Module):
    def __init__(self, clip_model, aux_model, sample_features, clip_weights, feat_dim, class_num, lambda_merge, alpha,
                 uncent_type, uncent_power):
        super().__init__()
        self.clip_model = clip_model
        self.aux_model = aux_model
        self.clip_weights = clip_weights
        self.aux_adapter = Linear_Adapter(feat_dim, class_num, sample_features=sample_features)

        self.lambda_merge = lambda_merge
        self.uncent_type = uncent_type
        self.uncent_power = uncent_power
        self.alpha = alpha

    def execute(self, images=None, clip_features=None, aux_features=None, labels=None):
        with jt.enable_grad():
            if images is not None:
                clip_features, aux_features = self.forward_feature(images)
            # print(-2)
            clip_features /= clip_features.norm(dim=-1, keepdim=True)
            aux_features /= aux_features.norm(dim=-1, keepdim=True)
            # print(-3)
            # clip_features.detach_inplace()
            # aux_features.detach_inplace()
            # print(clip_features.dtype)
            # print("features is_stop_grad:")
            # print(type(clip_features))
            # print(clip_features.is_stop_grad(), aux_features.is_stop_grad())
            clip_logits, aux_logits,aux_logits1 = self.forward_adapter(clip_features, aux_features)
            # clip_logits.start_grad(), aux_logits.start_grad()
            # print("logits is_stop_grad:")
            # print(clip_logits.is_stop_grad(), aux_logits.is_stop_grad())
            # print(-4)
            # fusion
            factor = uncertainty(
                clip_logits.float(),
                power=self.uncent_power,
                type=self.uncent_type
            )
            # print(-5)
            logits = clip_logits + factor * aux_logits * self.alpha

            # loss
            if labels is not None:
                # print(logits.is_stop_grad(),labels.is_stop_grad())  True True
                # loss_merge = F.cross_entropy(logits, labels)
                loss_merge = nn.cross_entropy_loss(logits, labels)
                # loss_aux = F.cross_entropy(aux_logits, labels)
                loss_aux = nn.cross_entropy_loss(aux_logits, labels)
                # print("loss_merge:{},loss_aux:{}".format(loss_merge, loss_aux))
                # print("lambda_merge:{}".format(self.lambda_merge))
                # print(loss_merge.is_stop_grad(),loss_aux.is_stop_grad())
                loss = self.lambda_merge * loss_merge + (1 - self.lambda_merge) * loss_aux
                # print("函数内部梯度:", jt.grad(loss, [loss_merge, loss_aux,logits,clip_logits,aux_logits,clip_features,aux_logits1]))
            else:
                loss_merge = None
                loss_aux = None
                loss = None

            return_dict = {
                "logits": logits,
                "clip_logits": clip_logits,
                "aux_logits": aux_logits,
                "loss": loss,
                "loss_merge": loss_merge,
                "loss_aux": loss_aux,
            }

            return return_dict

    def forward_feature(self, images):
        # CLIP branch
        clip_features = self.clip_model.encode_image(tfm_clip(images))
        # AUX branch
        aux_feature = self.aux_model(tfm_aux(images))
        return clip_features, aux_feature

    def forward_adapter(self, clip_features, aux_features):
        # logits
        clip_logits = 100. * clip_features @ self.clip_weights

        aux_logits1 = self.aux_adapter(aux_features)
        aux_logits = logit_normalize(aux_logits1)
        return clip_logits, aux_logits,aux_logits1 