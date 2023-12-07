import torchvision
import torch as th
import torch.nn as nn

""" VGG-16 Structure
Input image is [-1, 3, 224, 224]
----------------------------------------------------------------------------------------------
        Layer (type)               Output Shape         Param #     Layer index   Conv. Index
==============================================================================================
            Conv2d-1         [-1, 64, 224, 224]           1,792               0             0
              ReLU-2         [-1, 64, 224, 224]               0               1
            Conv2d-3         [-1, 64, 224, 224]          36,928               2             1
              ReLU-4         [-1, 64, 224, 224]               0               3
         MaxPool2d-5         [-1, 64, 112, 112]               0               4
            Conv2d-6        [-1, 128, 112, 112]          73,856               5             2
              ReLU-7        [-1, 128, 112, 112]               0               6
            Conv2d-8        [-1, 128, 112, 112]         147,584               7             3
              ReLU-9        [-1, 128, 112, 112]               0               8
        MaxPool2d-10          [-1, 128, 56, 56]               0               9
           Conv2d-11          [-1, 256, 56, 56]         295,168              10             4
             ReLU-12          [-1, 256, 56, 56]               0              11
           Conv2d-13          [-1, 256, 56, 56]         590,080              12             5
             ReLU-14          [-1, 256, 56, 56]               0              13
           Conv2d-15          [-1, 256, 56, 56]         590,080              14             6
             ReLU-16          [-1, 256, 56, 56]               0              15
        MaxPool2d-17          [-1, 256, 28, 28]               0              16
           Conv2d-18          [-1, 512, 28, 28]       1,180,160              17             7
             ReLU-19          [-1, 512, 28, 28]               0              18
           Conv2d-20          [-1, 512, 28, 28]       2,359,808              19             8
             ReLU-21          [-1, 512, 28, 28]               0              20
           Conv2d-22          [-1, 512, 28, 28]       2,359,808              21             9     
             ReLU-23          [-1, 512, 28, 28]               0              22
        MaxPool2d-24          [-1, 512, 14, 14]               0              23
           Conv2d-25          [-1, 512, 14, 14]       2,359,808              24            10
             ReLU-26          [-1, 512, 14, 14]               0              25
           Conv2d-27          [-1, 512, 14, 14]       2,359,808              26            11
             ReLU-28          [-1, 512, 14, 14]               0              27
           Conv2d-29          [-1, 512, 14, 14]       2,359,808              28            12
             ReLU-30          [-1, 512, 14, 14]               0              29
        MaxPool2d-31            [-1, 512, 7, 7]               0    
==============================================================================================
"""

class SemanticEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = torchvision.models.vgg16(pretrained=True).eval().cuda().features
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])

    def encode_feats(self, img: th.Tensor, layers=[11, 13, 15], size=(256, 256)) -> th.Tensor:
        # get to the gpu
        if not img.is_cuda:
            img = img.cuda()

        i = self.normalize(img)
        if size is not None:
            resize = torchvision.transforms.Resize(size)
            i = resize(i)
        outputs = []

        for ix, layer in enumerate(self.backbone):
            if ix > max(layers):
                break

            i = layer(i)
            if ix in layers:
                outputs.append(i)

        feat = th.stack(outputs, dim=0)

        return feat

    def gram_loss(self, content, style):
        a, b, c = content.size()
        G_c = th.einsum('abc,efg->bf', content, content).div(a * b * c)
        G_s = th.einsum('abc,efg->bf', style, style).div(a * b * c)
        return th.pow(G_c - G_s, 2).mean()

    # Modified from Ref-NPR
    # https://ref-npr.github.io/
    def nn_feat_replace(self, content, content_style, style):
        content = content.flatten(-2, -1)
        content_style = content_style.flatten(-2, -1)
        style = style.flatten(-2, -1)

        n, c, hw = content.size()

        z_new = []
        for i in range(n):
            z_best = self.argmin_cos_distance(content[i: i + 1], content_style[i: i + 1])
            z_best = z_best.unsqueeze(1).repeat(1, c, 1)
            feat = th.gather(style[i: i + 1], 2, z_best)
            z_new.append(feat)

        z_new = th.cat(z_new, 0)
        # z_new = z_new.view(n, c, hw)
        return z_new

    def nn_feat_replace_color(self, content, content_style, style_color):
        _, h, w = style_color.size()

        content = content.flatten(-2, -1)
        content_style = content_style.flatten(-2, -1)
        style_color = style_color.flatten(-2, -1)

        n, c, hw = content.size()
        c_, _ = style_color.size()

        c_new = []
        d_new = []
        for i in range(n):
            z_best, d_best = self.cos_distance(content[i: i + 1], content_style[i: i + 1])
            c_best = z_best.repeat(c_, 1)
            color = th.gather(style_color, 1, c_best)
            c_new.append(color)
            d_new.append(d_best)

        best_d = th.argmin(th.cat(d_new, 0), dim=0)
        c_new = th.stack(c_new)
        c_new = th.gather(c_new, 0, best_d.repeat(3, 1)[None, ...]).squeeze()
        c_new = c_new.view(-1, h, w)
        return c_new

    # Modified from Ref-NPR
    # https://ref-npr.github.io/
    def cos_loss(self, a, b):
        a_norm = (a * a).sum(1, keepdims=True).sqrt()
        b_norm = (b * b).sum(1, keepdims=True).sqrt()
        a_tmp = a / (a_norm + 1e-8)
        b_tmp = b / (b_norm + 1e-8)
        cossim = (a_tmp * b_tmp).sum(1)
        cos_d = 1.0 - cossim
        return cos_d.mean()

    # Modified from Ref-NPR
    # https://ref-npr.github.io/
    def argmin_cos_distance(self, a, b, center=False):
        """
        a: [b, c, hw],
        b: [b, c, h2w2]
        """
        if center:
            a = a - a.mean(2, keepdims=True)
            b = b - b.mean(2, keepdims=True)

        b_norm = ((b * b).sum(1, keepdims=True) + 1e-8).sqrt()
        b = b / (b_norm + 1e-8)

        z_best = []
        loop_batch_size = int(1e8 / b.shape[-1])
        for i in range(0, a.shape[-1], loop_batch_size):
            a_batch = a[..., i: i + loop_batch_size]
            a_batch_norm = ((a_batch * a_batch).sum(1, keepdims=True) + 1e-8).sqrt()
            a_batch = a_batch / (a_batch_norm + 1e-8)

            d_mat = 1.0 - th.matmul(a_batch.transpose(2, 1), b)

            z_best_batch = th.argmin(d_mat, 2)
            z_best.append(z_best_batch)
        z_best = th.cat(z_best, dim=-1)

        return z_best

    def cos_distance(self, a, b, center=False):
        """
        a: [b, c, hw],
        b: [b, c, h2w2]
        """
        if center:
            a = a - a.mean(2, keepdims=True)
            b = b - b.mean(2, keepdims=True)

        b_norm = ((b * b).sum(1, keepdims=True) + 1e-8).sqrt()
        b = b / (b_norm + 1e-8)

        z_best = []
        d_best = []
        loop_batch_size = int(1e8 / b.shape[-1])
        for i in range(0, a.shape[-1], loop_batch_size):
            a_batch = a[..., i: i + loop_batch_size]
            a_batch_norm = ((a_batch * a_batch).sum(1, keepdims=True) + 1e-8).sqrt()
            a_batch = a_batch / (a_batch_norm + 1e-8)

            d_mat = 1.0 - th.matmul(a_batch.transpose(2, 1), b)

            z_best_batch = th.argmin(d_mat, 2)
            d_best.append(th.gather(d_mat, 2, z_best_batch[None, ...])[0])
            z_best.append(z_best_batch)
        z_best = th.cat(z_best, dim=-1)
        d_best = th.cat(d_best, dim=-1)

        return z_best, d_best

    def get_mean_patch_color(self, img, size=(32, 32)):
        if not img.is_cuda:
            img = img.cuda()

        if size is not None:
            resize = torchvision.transforms.Resize(size=size)
            img = resize(img)

        return img


    # Modified from Ref-NPR
    # https://ref-npr.github.io/
    def match_colors_for_image_set(self, image, style_img):
        """
        image_set: [H, W, 3]
        style_img: [H, W, 3]
        """
        image_set = image.reshape(-1, 3)
        style_img = style_img.reshape(-1, 3).to(image_set.device)

        mu_c = image_set.mean(0, keepdim=True)
        mu_s = style_img.mean(0, keepdim=True)

        cov_c = th.matmul((image_set - mu_c).transpose(1, 0), image_set - mu_c) / float(image_set.size(0))
        cov_s = th.matmul((style_img - mu_s).transpose(1, 0), style_img - mu_s) / float(style_img.size(0))

        u_c, sig_c, _ = th.svd(cov_c)
        u_s, sig_s, _ = th.svd(cov_s)

        u_c_i = u_c.transpose(1, 0)
        u_s_i = u_s.transpose(1, 0)

        scl_c = th.diag(1.0 / th.sqrt(th.clamp(sig_c, 1e-8, 1e8)))
        scl_s = th.diag(th.sqrt(th.clamp(sig_s, 1e-8, 1e8)))

        tmp_mat = u_s @ scl_s @ u_s_i @ u_c @ scl_c @ u_c_i
        tmp_vec = mu_s.view(1, 3) - mu_c.view(1, 3) @ tmp_mat.T

        image_set = image_set @ tmp_mat.T + tmp_vec.view(1, 3)
        image_set = image_set.contiguous().clamp_(0.0, 1.0).reshape(image_set.shape)

        color_tf = th.eye(4).float().to(tmp_mat.device)
        color_tf[:3, :3] = tmp_mat
        color_tf[:3, 3:4] = tmp_vec.T
        return image_set, color_tf

    def match_color(self, style_img, target_img, eps=1e-5):
        # code adapted from
        # https://github.com/leongatys/NeuralImageSynthesis/blob/master/ExampleNotebooks/ColourControl.ipynb

        # this is actually the style image
        mu_t = style_img.mean(axis=(1, 2), keepdim=True)
        t = (style_img - mu_t).flatten(1, 2)
        Ct = t @ t.T / t.shape[1] + eps * th.eye(t.shape[0], device=t.device)

        if len(target_img.shape) == 2:
            target_img = target_img[..., None]

        # this is actually the target image
        mu_s = target_img.mean(axis=(1, 2), keepdim=True)
        s = (target_img - mu_s).flatten(1, 2)
        Cs = s @ s.T / s.shape[1] + eps * th.eye(s.shape[0], device=s.device)

        eva_t, eve_t = th.linalg.eigh(Ct)
        Qt = eve_t @ th.sqrt(th.diag(eva_t)) @ eve_t.T
        eva_s, eve_s = th.linalg.eigh(Cs)
        Qs = eve_s @ th.sqrt(th.diag(eva_s)) @ eve_s.T
        ts = Qs @ th.linalg.inv(Qt) @ t

        matched_img = ts.reshape(style_img.shape)
        matched_img += mu_s
        matched_img = th.clamp(matched_img, 0, 1)

        return matched_img