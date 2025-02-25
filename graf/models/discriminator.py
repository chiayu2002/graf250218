import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, imsize=64, hflip=False, num_classes=1, cond=True):
        super(Discriminator, self).__init__()
        self.nc = nc
        # assert(imsize==32 or imsize==64 or imsize==128)
        self.imsize = imsize
        self.hflip = hflip
        self.num_classes = num_classes

        SN = torch.nn.utils.spectral_norm
        IN = lambda x : nn.InstanceNorm2d(x)

        blocks = []
        if self.imsize==128:
            blocks += [
                # input is (nc) x 128 x 128
                SN(nn.Conv2d(nc, ndf//2, 4, 2, 1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
                # input is (ndf//2) x 64 x 64
                SN(nn.Conv2d(ndf//2, ndf, 4, 2, 1, bias=False)),
                IN(ndf),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                SN(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
                #nn.BatchNorm2d(ndf * 2),
                IN(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        elif self.imsize==64:
            blocks += [
                # input is (nc) x 64 x 64
                SN(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                SN(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
                #nn.BatchNorm2d(ndf * 2),
                IN(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        else:
            blocks += [
                # input is (nc) x 32 x 32
                nn.Conv2d(nc, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                # IN(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        blocks += [
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            # IN(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            # IN(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            # SN(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)),
            # nn.Sigmoid()
        ]
        self.conv_out = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        blocks = [x for x in blocks if x]
        self.main = nn.Sequential(*blocks)

        # 條件嵌入模塊
        if cond:
            self.condition_embedding = nn.Sequential(
                nn.Embedding(num_classes, ndf * 8),
                nn.LayerNorm(ndf * 8)
                )

    def forward(self, input, label, return_features=False):
        input = input[:, :self.nc]
        input = input.view(-1, self.imsize, self.imsize, self.nc).permute(0, 3, 1, 2)  # (BxN_samples)xC -> BxCxHxW

        first_label = label[:,0]
        first_label = first_label.long().to(input.device)
        
        label_embedding = self.condition_embedding(first_label)
        label_pred = label_embedding

        if self.hflip:      # Randomly flip input horizontally
            input_flipped = input.flip(3)
            mask = torch.randint(0, 2, (len(input),1, 1, 1)).bool().expand(-1, *input.shape[1:])
            input = torch.where(mask, input, input_flipped)
        a = self.main(input)
        label_embedding = label_embedding.view(label_embedding.size(0), -1, 1, 1)  # (B, n_feat, 1, 1)
        conditioned_features = a * label_embedding
        out = self.conv_out(conditioned_features)

        if return_features:
            return out, label_pred

        return out

import torch
import torch.nn as nn

# class Dvgg(nn.Module):
#     def __init__(self, num_classes):
#         super(Dvgg, self).__init__()
#         # VGG16的特徵提取層
#         self.features = nn.Sequential(
#             # Block 1
#             nn.Conv2d(3, 64, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, 2),
            
#             # Block 2
#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, 2),
            
#             # Block 3
#             nn.Conv2d(128, 256, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, 2),
            
#             # Block 4
#             nn.Conv2d(256, 512, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, 2),
            
#             # Block 5
#             nn.Conv2d(512, 512, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, 2)
#         )
        
#         # 分類器
#         self.classifier = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, num_classes)
#         )
        
#     def downsample(self, x, scale_factor=0.5):
#         """下採樣函數"""
#         return nn.functional.interpolate(
#             x,
#             scale_factor=scale_factor,
#             mode='bilinear',
#             align_corners=False
#         )
        
#     def forward(self, P_prime, I):
#         """
#         前向傳播
#         Args:
#             P_prime: 生成的圖像點 shape: [8192, 3]
#             I: 原始圖像 shape: [8, 3, 128, 128]
#         Returns:
#             P_prime_cls: P'的分類結果
#             I_cls: I的分類結果
#         """
#         batch_size = I.shape[0]  # 8
        
#         # 重塑 P_prime 為 [8, 3, 32, 32]
#         P_prime = P_prime.view(batch_size, 3, 32, 32)
        
#         # 調整輸入尺寸為 VGG16 預期的 224x224
#         P_prime = nn.functional.interpolate(P_prime, size=(224, 224), mode='bilinear', align_corners=False)
#         I_resized = nn.functional.interpolate(I, size=(224, 224), mode='bilinear', align_corners=False)
        
#         # 提取特徵
#         P_prime_features = self.features(P_prime)
#         I_features = self.features(I_resized)
        
#         # 展平特徵 - 現在應該是 [8, 512, 7, 7]
#         P_prime_flat = P_prime_features.view(P_prime_features.size(0), -1)  # [8, 25088]
#         I_flat = I_features.view(I_features.size(0), -1)  # [8, 25088]
        
#         # 分類
#         P_prime_cls = self.classifier(P_prime_flat)
#         I_cls = self.classifier(I_flat)
        
#         return P_prime_cls, I_cls