import torch
import torch.nn as nn
import init
from collections import OrderedDict
from correlation_package.correlation import Correlation

class Model(nn.Module):
    def __init__(self, inc, outc, corr=True, pretrained=True):
        super(Model, self).__init__()

        self.corr = corr

        # encoder1
        self.enc1_conv1 = nn.Conv2d(int(inc/2), 64, 7, padding=3, stride=2, bias=False)
        self.enc1_bn1   = nn.BatchNorm2d(64)
        self.enc1_pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.enc1_res1_1 = ResBL( 64,  64,  64, stride=1)
        self.enc1_res1_2 = ResBL( 64,  64,  64, stride=1)
        self.enc1_res2_1 = ResBL( 64, 128, 128, stride=2)
        self.enc1_res2_2 = ResBL(128, 128, 128, stride=1)
        self.enc1_res3_1 = ResBL(128, 256, 256, stride=2)
        self.enc1_res3_2 = ResBL(256, 256, 256, stride=1)
        self.enc1_res4_1 = ResBL(256, 512, 512, stride=2)
        self.enc1_res4_2 = ResBL(512, 512, 512, stride=1)
        self.enc1_conv5 = nn.Conv2d( 512, 1024, 3, padding=1, stride=2)
        self.enc1_bn5   = nn.BatchNorm2d(1024)
        self.enc1_conv6 = nn.Conv2d(1024, 1024, 3, padding=1, stride=1)
        self.enc1_bn6   = nn.BatchNorm2d(1024)

        # encoder2
        self.enc2_conv1 = nn.Conv2d(int(inc/2), 64, 7, padding=3, stride=2, bias=False)
        self.enc2_bn1   = nn.BatchNorm2d(64)
        self.enc2_pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.enc2_res1_1 = ResBL( 64,  64,  64, stride=1)
        self.enc2_res1_2 = ResBL( 64,  64,  64, stride=1)
        self.enc2_res2_1 = ResBL( 64, 128, 128, stride=2)
        self.enc2_res2_2 = ResBL(128, 128, 128, stride=1)
        self.enc2_res3_1 = ResBL(128, 256, 256, stride=2)
        self.enc2_res3_2 = ResBL(256, 256, 256, stride=1)
        self.enc2_res4_1 = ResBL(256, 512, 512, stride=2)
        self.enc2_res4_2 = ResBL(512, 512, 512, stride=1)
        self.enc2_conv5 = nn.Conv2d( 512, 1024, 3, padding=1, stride=2)
        self.enc2_bn5   = nn.BatchNorm2d(1024)
        self.enc2_conv6 = nn.Conv2d(1024, 1024, 3, padding=1, stride=1)
        self.enc2_bn6   = nn.BatchNorm2d(1024)

        # decoder
        self.dec_conv6 = nn.Conv2d(2048, 1024, 3, padding=1, stride=1)
        self.dec_bn6   = nn.BatchNorm2d(1024)
        self.dec_conv5 = nn.Conv2d(1024,  512, 3, padding=1, stride=1)
        self.dec_bn5   = nn.BatchNorm2d(512)
        self.dec_res4_2 = ResBL( 512, 512, 512, upscale=1, skip2=1024)
        self.dec_res4_1 = ResBL( 512, 512, 256, upscale=2)
        self.dec_res3_2 = ResBL( 256, 256, 256, upscale=1, skip2=512)
        self.dec_res3_1 = ResBL( 256, 256, 128, upscale=2)
        if self.corr is True:
            self.dec_corr2 = Correlation(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2, corr_multiply=1)
            self.dec_res2_2 = ResBL( 128, 128, 128, upscale=1, skip1=256+21*21)
        else:
            self.dec_res2_2 = ResBL(128, 128, 128, upscale=1, skip1=256)
        self.dec_res2_1 = ResBL( 128, 128,  64, upscale=2)
        self.dec_res1_2 = ResBL(  64,  64,  64, upscale=1, skip2=128)
        self.dec_res1_1 = ResBL(  64,  64,  64, upscale=1)
        if self.corr is True:
            self.dec_corr1 = Correlation(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2, corr_multiply=1)
            self.dec_conv1 = nn.Conv2d(192+21*21, 64, 7, padding=3, stride=1, bias=False)
        else:
            self.dec_conv1 = nn.Conv2d(192, 64, 7, padding=3, stride=1, bias=False)
        self.dec_bn1   = nn.BatchNorm2d(64)

        # classifier
        self.classifier = nn.Conv2d(64, outc, 1, padding=0, stride=1)

        # util
        self.unpool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.relu = nn.ReLU(inplace=True)
        if self.corr is True:
            self.corr_activation = nn.LeakyReLU(0.1,inplace=True)

        # initialization
        self.init_weights()
        if pretrained is True:
            self.load_net_param()

    def forward(self, x):
        x1, x2 = torch.split(x,3,1)

        # encoder1
        enc1_f1 = self.enc1_conv1(x1)
        enc1_f1 = self.enc1_bn1(enc1_f1)
        enc1_f1 = self.relu(enc1_f1)
        enc1_f2 = self.enc1_pool1(enc1_f1)
        enc1_f2 = self.enc1_res1_1(enc1_f2)
        enc1_f2 = self.enc1_res1_2(enc1_f2)
        enc1_f3 = self.enc1_res2_1(enc1_f2)
        enc1_f3 = self.enc1_res2_2(enc1_f3)
        enc1_f4 = self.enc1_res3_1(enc1_f3)
        enc1_f4 = self.enc1_res3_2(enc1_f4)
        enc1_f5 = self.enc1_res4_1(enc1_f4)
        enc1_f5 = self.enc1_res4_2(enc1_f5)
        enc1_f6 = self.enc1_conv5(enc1_f5)
        enc1_f6 = self.enc1_bn5(enc1_f6)
        enc1_f6 = self.relu(enc1_f6)
        enc1_f6 = self.enc1_conv6(enc1_f6)
        enc1_f6 = self.enc1_bn6(enc1_f6)
        enc1_f6 = self.relu(enc1_f6)

        # encoder2
        enc2_f1 = self.enc2_conv1(x2)
        enc2_f1 = self.enc2_bn1(enc2_f1)
        enc2_f1 = self.relu(enc2_f1)
        enc2_f2 = self.enc2_pool1(enc2_f1)
        enc2_f2 = self.enc2_res1_1(enc2_f2)
        enc2_f2 = self.enc2_res1_2(enc2_f2)
        enc2_f3 = self.enc2_res2_1(enc2_f2)
        enc2_f3 = self.enc2_res2_2(enc2_f3)
        enc2_f4 = self.enc2_res3_1(enc2_f3)
        enc2_f4 = self.enc2_res3_2(enc2_f4)
        enc2_f5 = self.enc2_res4_1(enc2_f4)
        enc2_f5 = self.enc2_res4_2(enc2_f5)
        enc2_f6 = self.enc2_conv5(enc2_f5)
        enc2_f6 = self.enc2_bn5(enc2_f6)
        enc2_f6 = self.relu(enc2_f6)
        enc2_f6 = self.enc2_conv6(enc2_f6)
        enc2_f6 = self.enc2_bn6(enc2_f6)
        enc2_f6 = self.relu(enc2_f6)

        # decoder
        enc_f6 = torch.cat([enc1_f6, enc2_f6], 1)
        dec = self.dec_conv6(enc_f6)
        dec = self.dec_bn6(dec)
        dec = self.relu(dec)
        dec = self.dec_conv5(dec)
        dec = self.unpool(dec)
        dec = self.dec_bn5(dec)
        dec = self.relu(dec)
        skp = torch.cat([enc1_f5, enc2_f5], 1)
        dec = self.dec_res4_2(dec, skip2=skp)
        dec = self.dec_res4_1(dec)
        skp = torch.cat([enc1_f4, enc2_f4], 1)
        dec = self.dec_res3_2(dec, skip2=skp)
        dec = self.dec_res3_1(dec)
        if self.corr is True:
            cor = self.dec_corr2(enc1_f3, enc2_f3)
            cor = self.corr_activation(cor)
            skp = torch.cat([enc1_f3, enc2_f3, cor], 1)
        else:
            skp = torch.cat([enc1_f3, enc2_f3], 1)
        dec = self.dec_res2_2(dec, skip1=skp)
        dec = self.dec_res2_1(dec)
        skp = torch.cat([enc1_f2, enc2_f2], 1)
        dec = self.dec_res1_2(dec, skip2=skp)
        dec = self.dec_res1_1(dec)
        dec = self.unpool(dec)
        if self.corr is True:
            cor = self.dec_corr1(enc1_f1, enc2_f1)
            cor = self.corr_activation(cor)
            dec = torch.cat([dec, enc1_f1, enc2_f1, cor], 1)
        else:
            dec = torch.cat([dec, enc1_f1, enc2_f1], 1)
        dec = self.dec_conv1(dec)
        dec = self.unpool(dec)
        dec = self.dec_bn1(dec)
        dec = self.relu(dec)

        out = self.classifier(dec)
        return out

    def init_weights(self):
        init.xavier_uniform_relu(self.modules())

    def load_net_param(self):
        from torchvision.models import resnet18
        resnet = resnet18(pretrained=True)

        self.enc1_conv1.load_state_dict(resnet.conv1.state_dict())
        self.enc1_bn1.load_state_dict(resnet.bn1.state_dict())
        self.enc1_res1_1.load_state_dict(list(resnet.layer1.children())[0].state_dict())
        self.enc1_res1_2.load_state_dict(list(resnet.layer1.children())[1].state_dict())
        self.enc1_res2_1.load_state_dict(list(resnet.layer2.children())[0].state_dict())
        self.enc1_res2_2.load_state_dict(list(resnet.layer2.children())[1].state_dict())
        self.enc1_res3_1.load_state_dict(list(resnet.layer3.children())[0].state_dict())
        self.enc1_res3_2.load_state_dict(list(resnet.layer3.children())[1].state_dict())
        self.enc1_res4_1.load_state_dict(list(resnet.layer4.children())[0].state_dict())
        self.enc1_res4_2.load_state_dict(list(resnet.layer4.children())[1].state_dict())

        self.enc2_conv1.load_state_dict(resnet.conv1.state_dict())
        self.enc2_bn1.load_state_dict(resnet.bn1.state_dict())
        self.enc2_res1_1.load_state_dict(list(resnet.layer1.children())[0].state_dict())
        self.enc2_res1_2.load_state_dict(list(resnet.layer1.children())[1].state_dict())
        self.enc2_res2_1.load_state_dict(list(resnet.layer2.children())[0].state_dict())
        self.enc2_res2_2.load_state_dict(list(resnet.layer2.children())[1].state_dict())
        self.enc2_res3_1.load_state_dict(list(resnet.layer3.children())[0].state_dict())
        self.enc2_res3_2.load_state_dict(list(resnet.layer3.children())[1].state_dict())
        self.enc2_res4_1.load_state_dict(list(resnet.layer4.children())[0].state_dict())
        self.enc2_res4_2.load_state_dict(list(resnet.layer4.children())[1].state_dict())


class ResBL(nn.Module):
    def __init__(self, inc, midc, outc, stride=1, upscale=1, skip1=0, skip2=0):
        super(ResBL, self).__init__()

        self.conv1 = nn.Conv2d(inc+skip1, midc, 3, padding=1, stride=stride, bias=False)
        self.bn1   = nn.BatchNorm2d(midc)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(midc+skip2, outc, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(outc)

        self.upscale = None
        if upscale > 1:
            self.upscale = nn.Upsample(scale_factor=upscale, mode='bilinear', align_corners=True)

        self.downsample = None

        if inc != outc or stride > 1 or upscale > 1:
            if upscale > 1:
                self.downsample = nn.Sequential(
                    nn.Conv2d(inc, outc, 1, padding=0, stride=stride, bias=False),
                    nn.Upsample(scale_factor=upscale, mode='bilinear', align_corners=True),
                    nn.BatchNorm2d(outc),
                )
            else:
                self.downsample = nn.Sequential(
                    nn.Conv2d(inc, outc, 1, padding=0, stride=stride, bias=False),
                    nn.BatchNorm2d(outc),
                )

    def forward(self, x, skip1=None, skip2=None):
        if skip1 is not None:
            res = torch.cat([x, skip1], 1)
        else:
            res = x

        res = self.conv1(res)
        res = self.bn1(res)
        res = self.relu(res)

        if skip2 is not None:
            res = torch.cat([res, skip2], 1)

        res = self.conv2(res)
        if self.upscale is not None:
            res = self.upscale(res)
        res = self.bn2(res)

        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        res += identity
        out = self.relu(res)

        return out




