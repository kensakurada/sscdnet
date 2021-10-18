import numpy as np
import cv2
import os.path
from argparse import ArgumentParser
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import sys
sys.path.append("./correlation_package/build/lib.linux-x86_64-3.6")
import cscdnet


class DataInfo:
    def __init__(self):
        self.width = 1024
        self.height = 224
        self.no_start = 0
        self.no_end = 100
        self.num_cv = 5

class Test:
    def __init__(self, arguments):
        self.args = arguments
        self.di = DataInfo()

    def test(self):

        _inputs = torch.from_numpy(np.concatenate((self.t0, self.t1), axis=0)).contiguous()
        _inputs = Variable(_inputs).view(1, -1, self.h_resize, self.w_resize)
        _inputs = _inputs.cuda()
        _outputs = self.model(_inputs)

        inputs = _inputs[0].cpu().data
        image_t0 = inputs[0:3, :, :]
        image_t1 = inputs[3:6, :, :]
        image_t0 = (image_t0 + 1.0) * 128
        image_t1 = (image_t1 + 1.0) * 128
        mask_gt = np.where(self.mask.data.numpy().squeeze(axis=0) == True, 0, 255)

        outputs = _outputs[0].cpu().data
        mask_pred = F.softmax(outputs[0:2, :, :], dim=0)[1] * 255

        self.display_results(image_t0, image_t1, mask_pred, mask_gt)

    def display_results(self, t0, t1, mask_pred, mask_gt):

        w, h = self.w_orig, self.h_orig
        t0_disp = cv2.resize(np.transpose(t0.numpy(), (1, 2, 0)).astype(np.uint8), (w, h))
        t1_disp = cv2.resize(np.transpose(t1.numpy(), (1, 2, 0)).astype(np.uint8), (w, h))
        mask_pred_disp = cv2.resize(cv2.cvtColor(mask_pred.numpy().astype(np.uint8), cv2.COLOR_GRAY2RGB), (w, h))
        mask_gt_disp = cv2.resize(cv2.cvtColor(mask_gt.astype(np.uint8), cv2.COLOR_GRAY2RGB), (w, h))

        img_out = np.zeros((h* 2, w * 2, 3), dtype=np.uint8)
        img_out[0:h, 0:w, :] = t0_disp
        img_out[0:h, w:w * 2, :] = t1_disp
        img_out[h:h * 2, 0:w * 1, :] = mask_gt_disp
        img_out[h:h * 2, w * 1:w * 2, :] = mask_pred_disp
        for dn, img in zip(['mask', 'disp'], [mask_pred_disp, img_out]):
            dn_save = os.path.join(self.args.checkpointdir, 'result', dn)
            fn_save = os.path.join(dn_save, '{0:08d}.png'.format(self.index))
            if not os.path.exists(dn_save):
                os.makedirs(dn_save)
            print('Writing ... ' + fn_save)
            cv2.imwrite(fn_save, img)

    def run(self):

        for i_set in range(0,self.di.num_cv):
            if self.args.use_corr:
                print('Correlated Siamese Change Detection Network (CSCDNet)')
                self.model = cscdnet.Model(inc=6, outc=2, corr=True, pretrained=True)
                fn_model = os.path.join(os.path.join(self.args.checkpointdir, 'set{}'.format(i_set), 'cscdnet-00030000.pth'))
            else:
                print('Siamese Change Detection Network (Siamese CDResNet)')
                self.model = cscdnet.Model(inc=6, outc=2, corr=False, pretrained=True)
                fn_model = os.path.join(os.path.join(self.args.checkpointdir, 'set{}'.format(i_set), 'cdnet-00030000.pth'))

            if os.path.isfile(fn_model) is False:
                print("Error: Cannot read file ... " + fn_model)
                exit(-1)
            else:
                print("Reading model ... " + fn_model)
            self.model.load_state_dict(torch.load(fn_model))
            self.model = self.model.cuda()

            if self.args.dataset == 'PCD':
                from dataset_pcd import PCD_full
                for dataset in ['TSUNAMI']:
                    loader_test = PCD_full(os.path.join(self.args.datadir,dataset), self.di.no_start, self.di.no_end, self.di.width, self.di.height)
                    for index in range(0,loader_test.__len__()):
                        if i_set * (10 / self.di.num_cv) <= (index % 10) < (i_set + 1) * (10 / self.di.num_cv):
                            self.index = index
                            self.t0, self.t1, self.mask, self.w_orig, self.h_orig, self.w_resize, self.h_resize = loader_test.__getitem__(index)
                            self.test()
                        else:
                            continue
            else:
                print('Error: Unexpected dataset')
                exit(-1)


if __name__ == '__main__':

    parser = ArgumentParser(description='Start testing ...')
    parser.add_argument('--datadir', required=True)
    parser.add_argument('--checkpointdir', required=True)
    parser.add_argument('--use-corr', action='store_true', help='using correlation layer')
    parser.add_argument('--dataset', required=True)

    test = Test(parser.parse_args())
    test.run()


