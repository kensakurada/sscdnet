
from __future__ import print_function
from argparse import ArgumentParser
import cv2
import csv
import os.path
import numpy as np
import torch
from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from criterion import CrossEntropyLoss2d
from dataset_pcd import PCD
import sys
sys.path.append("./correlation_package/build/lib.linux-x86_64-3.6")
import cscdnet


def colormap():
    cmap=np.zeros([2, 3]).astype(np.uint8)

    cmap[0,:] = np.array([0, 0, 0])
    cmap[1,:] = np.array([255, 255, 255])

    return cmap


class Colorization:

    def __init__(self, n=2):
        self.cmap = colormap()
        self.cmap = torch.from_numpy(np.array(self.cmap[:n]))

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = gray_image[0] == label

            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


class Training:
    def __init__(self, arguments):
        self.args = arguments
        self.icount = 0
        if self.args.use_corr:
            self.dn_save = os.path.join(self.args.checkpointdir,'cscdnet','checkpointdir','set{}'.format(self.args.cvset))
        else:
            self.dn_save = os.path.join(self.args.checkpointdir,'cdnet','checkpointdir','set{}'.format(self.args.cvset))

    def train(self):

        self.color_transform = Colorization(2)

        # Dataset loader for train and test
        dataset_train = DataLoader(
            PCD(os.path.join(self.args.datadir, 'set{}'.format(self.args.cvset), 'train')),
            num_workers=self.args.num_workers, batch_size=self.args.batch_size, shuffle=True)
        self.dataset_test = PCD(os.path.join(self.args.datadir, 'set{}'.format(self.args.cvset), 'test'))

        self.test_path = os.path.join(self.dn_save, 'test')
        if not os.path.exists(self.test_path):
            os.makedirs(self.test_path)

        # Set loss function, optimizer and learning rate
        weight = torch.ones(2)
        criterion = CrossEntropyLoss2d(weight.cuda())
        optimizer = Adam(self.model.parameters(), lr=0.0001, betas=(0.5, 0.999))
        lambda1 = lambda icount: (float)(self.args.max_iteration - icount) / (float)(self.args.max_iteration)
        model_lr_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

        fn_loss = os.path.join(self.dn_save,'loss.csv')
        f_loss =  open(fn_loss, 'w')
        writer = csv.writer(f_loss)

        self.writers= SummaryWriter(os.path.join(self.dn_save, 'log'))

        # Training loop
        icount_loss = []
        while self.icount < self.args.max_iteration:
            for step, (inputs_train, mask_train) in enumerate(dataset_train):
                inputs_train = inputs_train.cuda()
                mask_train = mask_train.cuda()

                inputs_train = Variable(inputs_train)
                mask_train = Variable(mask_train)
                outputs_train = self.model(inputs_train)

                optimizer.zero_grad()
                self.loss = criterion(outputs_train, mask_train[:, 0])

                self.loss.backward()
                optimizer.step()

                self.icount += 1
                icount_loss.append(self.loss.item())
                writer.writerow([self.icount, self.loss.item()])
                if self.args.icount_plot > 0 and self.icount % self.args.icount_plot == 0:
                    self.test()
                    average = sum(icount_loss) / len(icount_loss)
                    print('loss: {0} (icount: {1})'.format(average, self.icount))
                    icount_loss.clear()

                if self.args.icount_save > 0 and self.icount % self.args.icount_save == 0:
                    self.checkpoint()

            # Call lr_schduler.step() after optimizer.step()
            model_lr_scheduler.step()

        f_loss.close()

    def test(self):

        index_test = self.dataset_test.get_random_index()
        inputs_test, mask_gt_test = self.dataset_test[index_test]
        inputs_test = inputs_test[np.newaxis, :, :]
        inputs_test = inputs_test.cuda()
        inputs_test = Variable(inputs_test)
        outputs_test = self.model(inputs_test)

        inputs = inputs_test[0].cpu().data
        t0_test = inputs[0:3, :, :]
        t1_test = inputs[3:6, :, :]
        t0_test = (t0_test + 1.0) * 128
        t1_test = (t1_test + 1.0) * 128
        mask_gt = mask_gt_test.numpy().astype(np.uint8) * 255

        outputs = outputs_test[0][np.newaxis, :, :, :]
        outputs = outputs[:, 0:2, :, :]
        mask_pred = np.transpose(self.color_transform(outputs[0].cpu().max(0)[1][np.newaxis, :, :].data).numpy(), (1, 2, 0)).astype(np.uint8)

        img_out = self.display_results(t0_test, t1_test, mask_pred, mask_gt)
        self.log_tbx(torch.from_numpy(np.transpose(np.flip(img_out,axis=2).copy(), (2, 0, 1))))

    def display_results(self, t0, t1, mask_pred, mask_gt):

        rows = cols = 256
        img_out = np.zeros((rows * 2, cols * 2, 3), dtype=np.uint8)
        img_out[0:rows, 0:cols, :] = np.transpose(t0.numpy(), (1, 2, 0)).astype(np.uint8)
        img_out[0:rows, cols:cols * 2, :] = np.transpose(t1.numpy(), (1, 2, 0)).astype(np.uint8)
        img_out[rows:rows * 2, 0:cols, :] = cv2.cvtColor(np.transpose(mask_gt, (1, 2, 0)), cv2.COLOR_GRAY2RGB)
        img_out[rows:rows * 2, cols:cols * 2, :] = mask_pred

        return img_out

    # Output results for tensorboard
    def log_tbx(self, image):

        writer = self.writers
        writer.add_scalar('data/loss', self.loss.item(), self.icount)
        writer.add_image('change detection', image, self.icount)

    def checkpoint(self):
        if self.args.use_corr:
            filename = 'cscdnet-{0:08d}.pth'.format(self.icount)
        else:
            filename = 'cdnet-{0:08d}.pth'.format(self.icount)
        
        # Enable generic saving of module if using multiple GPU's
        if torch.cuda.device_count() > 1:
            torch.save(self.model.module.state_dict(), os.path.join(self.dn_save, filename))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.dn_save, filename))
                
        print('save: {0} (iteration: {1})'.format(filename, self.icount))

    def run(self):

        if self.args.use_corr:
            print('Correlated Siamese Change Detection Network (CSCDNet)')
            self.model = cscdnet.Model(inc=6, outc=2, corr=True, pretrained=True)
        else:
            print('Siamese Change Detection Network (Siamese CDResNet)')
            self.model = cscdnet.Model(inc=6, outc=2, corr=False, pretrained=True)

        # Run on muliple GPU's if available
        if torch.cuda.device_count() > 1:
            print("Training with", torch.cuda.device_count(), "GPU's")
            model = torch.nn.DataParallel(model)

        self.model = self.model.cuda()
        self.train()


if __name__ == '__main__':

    parser = ArgumentParser(description='Start training ...')
    parser.add_argument('--checkpointdir', required=True)
    parser.add_argument('--datadir', required=True)
    parser.add_argument('--use-corr', action='store_true', help='using correlation layer')
    parser.add_argument('--max-iteration', type=int, default=50000)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--cvset', type=int, default=0)
    parser.add_argument('--icount-plot', type=int, default=0)
    parser.add_argument('--icount-save', type=int, default=10)

    training = Training(parser.parse_args())
    training.run()
