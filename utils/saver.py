import os
import torch
import glob


class Saver(object):
    def __init__(self, args):
        self.args = args
        self.directory = os.path.join('run', args.dataset, args.checkname)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, swa=False):
        """Saves checkpoint to disk"""
        if swa:
            filename = os.path.join(self.experiment_dir, 'swa_%03d_checkpoint.pth.tar' % state['epoch'])
        else:
            filename = os.path.join(self.experiment_dir, '%03d_checkpoint.pth.tar' % (state['epoch']))

        torch.save(state, filename)



