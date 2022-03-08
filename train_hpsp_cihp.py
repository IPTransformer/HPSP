import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
import time
import torch.backends.cudnn as cudnn
from pytorch_toolbelt import losses as L

from utils.losses import DiceLoss, SoftCrossEntropyLoss, LovaszLoss
from utils.util import reduce_mean, AverageMeter, ProgressMeter, intersectionAndUnionGPU, inial_logger
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.matcher import HungarianMatcher
from dataset.batch_sampler import BatchSampler
from utils.losses.human import HumanLoss
from dataset import build_dataset
from models import build_cihp_model

# from models.iptr_cihp import IPTR
# from models.resnet50_cihp import main_model as IPTR
import random



try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(111)
print("setting seed")


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=100, power=1.0):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    if iter % lr_decay_iter or iter > max_iter:
        return optimizer

    lr = init_lr * (1 - iter / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


class Trainer(object):
    def __init__(self, args):
        args.nprocs = torch.cuda.device_count()
        args.batch_size = int(args.batch_size / args.nprocs)
        self.args = args

        if args.local_rank == 0:
            self.saver = Saver(args)
            self.summary = TensorboardSummary(self.saver.experiment_dir)
            self.writer = self.summary.create_summary()
            self.logger = inial_logger(os.path.join(self.saver.experiment_dir, 'log.log'))
            self.logger.info(self.args)
            self.logger.info('Starting Epoch:{}'.format(self.args.start_epoch))
            self.logger.info('Total Epoches:{}'.format(self.args.epochs))

        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group('nccl', init_method='env://')

        self.device = torch.device('cuda:{}'.format(args.local_rank))

        # define model
        self.nclass = args.nclass
        self.ninstance = args.ninstance
        # model = IPTR(backbone=args.backbone, num_classes=self.nclass, num_instances=self.ninstance, use_checkpoint=False)
        build_model = build_cihp_model(args.backbone)
        model = build_model(backbone=args.backbone, num_classes=self.nclass, num_instances=self.ninstance, use_checkpoint=False)

        import apex
        print("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)
        model = model.cuda().to(memory_format=torch.contiguous_format)


        self.model = model
        if self.args.opt == 'adam':
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-4)

        else:
            self.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                             weight_decay=args.weight_decay, nesterov=args.nesterov)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=self.args.T0,
                                                                                  T_mult=self.args.Tm, last_epoch=-1)

        self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
                                                    opt_level='O1')
        if args.nprocs > 1:
            self.model = DDP(self.model, delay_allreduce=True)


        cudnn.benchmark = True

        # from dataset.cihp import CIHPDataset
        Dataset = build_dataset(args.dataset)
        img_size = [args.img_size, args.img_size]

        train_set = Dataset(args, mode='train', img_size=img_size)
        val_set = Dataset(args, mode='val', img_size=img_size)

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        self.train_sampler = train_sampler
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
        kwargs = {'num_workers': args.workers, 'pin_memory': False}
        self.train_loader = DataLoader(train_set,
                                       batch_sampler=BatchSampler(train_sampler,
                                                                  batch_size=args.batch_size,
                                                                  drop_last=True,
                                                                  multiscale_step=1,
                                                                  img_sizes=args.base_size,
                                                                  rms=False),
                                       **kwargs)
        self.val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                                     sampler=val_sampler, **kwargs)

        self.total_inter = args.epochs * len(self.train_loader)
        # Define Criterion
        if self.args.loss_type == 'dice':
            DiceLoss_fn = DiceLoss(mode='multiclass', ignore_index=255)
        else:
            DiceLoss_fn = LovaszLoss(mode='multiclass', per_image=False, ignore_index=255)

        SoftCrossEntropy_fn = SoftCrossEntropyLoss(smooth_factor=0.1, ignore_index=255)
        self.criterion1 = L.JointLoss(first=DiceLoss_fn, second=SoftCrossEntropy_fn,
                                      first_weight=0.5, second_weight=0.5).cuda()
        self.criterion2 = HumanLoss().cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])

        self.iteration = 0
        self.matcher = HungarianMatcher().cuda()
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx


    def compute_loss_basedon_matcher(self, pred, target, criterion):
        # ----------------------Use HungarianMatcher--------------------------
        # N, C, H, W = pred.shape

        indices = self.matcher(pred.detach().clone().float(), target.detach().clone().float())

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = pred[src_idx]  # C_new x H x W
        tgt_masks = target[tgt_idx]  # C_new x H x W
        loss = criterion(src_masks, tgt_masks)
        return loss



    def training(self, epoch):
        train_loader = self.train_loader
        train_sampler = self.train_sampler

        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        losses_ins = AverageMeter('Loss_ins', ':.4e')
        losses_seg = AverageMeter('Loss_seg', ':.4e')
        losses_cls = AverageMeter('Loss_cls', ':.4e')
        LR = AverageMeter('LR', ':8.8f')
        progress = ProgressMeter(len(train_loader), [LR, batch_time, data_time,losses_ins, losses_cls, losses_seg, losses],
                                 prefix="Epoch: [{}]".format(epoch))

        train_sampler.set_epoch(epoch)
        self.model.train()

        end = time.time()
        for i, sample in enumerate(train_loader):
            data_time.update(time.time() - end)

            image, mask_target, ins_target = sample['image'], sample['catg'], sample['human_correct']
            image, mask_target, ins_target = image.to(self.device), mask_target.to(self.device), ins_target.to(
                self.device)
            poly_lr_scheduler(self.optimizer, init_lr=self.args.lr, iter=self.iteration, lr_decay_iter=1,
                              max_iter=self.total_inter)

            pred_ins, pred_mask = self.model(image)
            loss_mask = self.criterion1(pred_mask, mask_target.long())

            # ----------------------Use HungarianMatcher--------------------------
            loss_ins = self.compute_loss_basedon_matcher(pred_ins, ins_target.float(), self.criterion2)
            loss = loss_mask + loss_ins*0.1

            # torch.distributed.barrier()


            self.optimizer.zero_grad()
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            self.optimizer.step()



            # torch.distributed.barrier()
            reduced_loss = reduce_mean(loss, self.args.nprocs)
            losses.update(reduced_loss.item(), image.size(0))
            reduced_loss_seg = reduce_mean(loss_mask, self.args.nprocs)
            losses_seg.update(reduced_loss_seg.item(), image.size(0))
            reduced_loss_ins = reduce_mean(loss_ins, self.args.nprocs)
            losses_ins.update(reduced_loss_ins.item(), image.size(0))
            # reduced_loss_cls = reduce_mean(loss_cls, self.args.nprocs)
            # losses_cls.update(reduced_loss_cls.item(), image.size(0))


            torch.cuda.synchronize()

            # self.scheduler.step(epoch + i / len(train_loader))
            self.iteration += 1
            LR.update(self.optimizer.param_groups[-1]['lr'])

            batch_time.update(time.time() - end)
            end = time.time()

            if self.args.local_rank == 0 and i % 50 == 0:
                progress.display(i, self.logger)

        if self.args.local_rank == 0:
            self.writer.add_scalar('train/total_loss_epoch', losses.val, epoch)

    def validation(self, epoch, swa=False):
        losses = AverageMeter('Loss', ':.4e')
        losses_ins = AverageMeter('Loss_ins', ':.4e')
        losses_seg = AverageMeter('Loss_seg', ':.4e')

        intersection1_meter = AverageMeter('I', ':6.2f')
        union1_meter = AverageMeter('U', ':6.2f')
        target1_meter = AverageMeter('Target', ':6.2f')

        self.model.eval()

        for i, sample in enumerate(self.val_loader):
            image, mask_target, ins_target = sample['image'], sample['catg'], sample[
                'human_correct']
            image, mask_target,  ins_target = image.to(self.device), mask_target.to(
                self.device),  ins_target.to(self.device)


            with torch.no_grad():
                pred_ins, pred_mask = self.model(image)

                loss_mask = self.criterion1(pred_mask, mask_target.long())

                # ----------------------Use HungarianMatcher--------------------------
                loss_ins = self.compute_loss_basedon_matcher(pred_ins, ins_target.float(), self.criterion2)
                loss = loss_mask + loss_ins*0.1

            # torch.distributed.barrier()
            torch.cuda.synchronize()

            reduced_loss = reduce_mean(loss, self.args.nprocs)
            losses.update(reduced_loss.item(), image.size(0))
            reduced_loss_seg = reduce_mean(loss_mask, self.args.nprocs)
            losses_seg.update(reduced_loss_seg.item(), image.size(0))
            reduced_loss_ins = reduce_mean(loss_ins, self.args.nprocs)
            losses_ins.update(reduced_loss_ins.item(), image.size(0))



            output = pred_mask.max(1)[1]
            intersection1, union1, target1 = intersectionAndUnionGPU(output, mask_target, self.nclass, 255)

            dist.all_reduce(intersection1), dist.all_reduce(union1), dist.all_reduce(target1)
            intersection1, union1, target1 = intersection1.cpu().numpy(), union1.cpu().numpy(), target1.cpu().numpy()
            intersection1_meter.update(intersection1), union1_meter.update(union1), target1_meter.update(target1)

            iou1_class = intersection1_meter.sum / (union1_meter.sum + 1e-10)
            accuracy1_class = intersection1_meter.sum / (target1_meter.sum + 1e-10)
            # print(iou_class)
            mIoU1 = np.mean(iou1_class)
            mAcc1 = np.mean(accuracy1_class)
            allAcc1 = sum(intersection1_meter.sum) / (sum(target1_meter.sum) + 1e-10)

        if self.args.local_rank == 0:
            self.logger.info(
                "EPOCH:{}, Acc1:{}, Acc1_class:{}, mIoU1:{}, loss_ins:{}, loss_seg:{}, total_loss_epoch:{}".format
                (epoch, mAcc1, allAcc1, mIoU1, losses_ins.val, losses_seg.val, losses.val))
            self.writer.add_scalar('val/total_loss_epoch', losses.val, epoch)
            self.writer.add_scalar('val/mIoU', mIoU1, epoch)
            self.writer.add_scalar('val/Acc', mAcc1, epoch)
            self.writer.add_scalar('val/Acc_class', allAcc1, epoch)


            if True:  # (epoch > 110) or (epoch < 13):
                self.saver.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.module.state_dict()}, swa)



def main():
    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training")
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--backbone', type=str, default='Swin_L_w12',
                        help='backbone name (default: resnet)')
    parser.add_argument('--dataset', type=str, default='cihp',
                        help='dataset name (default: pascal)')
    parser.add_argument('--img-size', type=int, default=512,
                        help='img size to bre trained')
    parser.add_argument('--nclass', type=int, default=20,
                        help='img size to bre trained')
    parser.add_argument('--rotate', type=int, default=40,
                        help='img size to bre trained')
    parser.add_argument('--ninstance', type=int, default=12,
                        help='img size to bre trained')
    parser.add_argument('--datatype', type=str, default='rgb',
                        help='dataset name (default: pascal)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', nargs='+', type=int, default=256,
                        help='base image size')
    parser.add_argument('--loss-type', type=str, default='dice',
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=120, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=16,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=16,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='cos',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--opt', type=str, default='adam',
                        choices=['sgd', 'adam'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    parser.add_argument('--split', type=str, default='trainval',
                        help='set the checkpoint name')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    if torch.cuda.device_count() > 1:
        args.sync_bn = True
    else:
        args.sync_bn = False

    if args.checkname is None:
        args.checkname = '%s-%s' % (
            args.backbone,
            str(args.epochs))

        import datetime
        args.checkname = '%s-%s' % (args.checkname, datetime.datetime.now().strftime('%Y%m%d%H%M%S'))

        if args.resume is not None:
            res = args.resume.split('/experiment')
            args.checkname = os.path.basename(res[0])

    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if epoch < (args.epochs-10) and epoch%20==0:
            trainer.validation(epoch)
        if epoch > (args.epochs-10) and epoch % 3 == 0:
            trainer.validation(epoch)

    if args.local_rank == 0:
        trainer.writer.close()



if __name__ == "__main__":
    main()
