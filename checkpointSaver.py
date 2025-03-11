import os
import shutil
import torch
import glob
import operator
import logging
from timm.utils.model import unwrap_model, get_state_dict

_logger = logging.getLogger(__name__)

class CheckpointSaver:
    def __init__(
            self,
            model,
            optimizer,
            args=None,
            model_ema=None,
            amp_scaler=None,
            checkpoint_prefix='checkpoint',
            recovery_prefix='recovery',
            checkpoint_dir='',
            recovery_dir='',
            decreasing=False,
            max_history=20,
            unwrap_fn=unwrap_model,
            # save_interval=10  # Save every 10 epochs
    ):
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.model_ema = model_ema
        self.amp_scaler = amp_scaler
        
        self.checkpoint_files = []  # List of (filename, metric) tuples
        self.best_epoch = None
        self.best_metric = None
        self.curr_recovery_file = ''
        self.prev_recovery_file = ''
        self.can_hardlink = True

        self.checkpoint_dir = checkpoint_dir
        self.recovery_dir = recovery_dir
        self.save_prefix = checkpoint_prefix
        self.recovery_prefix = recovery_prefix
        self.extension = '.pth.tar'
        self.decreasing = decreasing  # Lower metric is better if True
        self.cmp = operator.lt if decreasing else operator.gt
        self.max_history = max_history
        self.unwrap_fn = unwrap_fn
        # self.save_interval = save_interval  # Save checkpoint every N epochs
        assert self.max_history >= 1

    def _replace(self, src, dst):
        if self.can_hardlink:
            try:
                if os.path.exists(dst):
                    os.unlink(dst)
            except (OSError, NotImplementedError):
                self.can_hardlink = False
        os.replace(src, dst)

    def _duplicate(self, src, dst):
        if self.can_hardlink:
            try:
                if os.path.exists(dst):
                    os.unlink(dst)
                os.link(src, dst)
                return
            except (OSError, NotImplementedError):
                self.can_hardlink = False
        shutil.copy2(src, dst)

    def _save(self, save_path, epoch, metric=None):
        save_state = {
            'epoch': epoch,
            'arch': type(self.model).__name__.lower(),
            'state_dict': get_state_dict(self.model, self.unwrap_fn),
            'optimizer': self.optimizer.state_dict(),
            'version': 2,
        }
        if self.args is not None:
            save_state['arch'] = self.args.model
            save_state['args'] = self.args
        if self.amp_scaler is not None:
            save_state[self.amp_scaler.state_dict_key] = self.amp_scaler.state_dict()
        if self.model_ema is not None:
            save_state['state_dict_ema'] = get_state_dict(self.model_ema, self.unwrap_fn)
        if metric is not None:
            save_state['metric'] = metric
        torch.save(save_state, save_path)

    def save_checkpoint(self, epoch=None, metric=None):
        if epoch is not None:
            assert epoch >= 0

        # if epoch is not None and epoch % self.save_interval != 0:
        #     return self.best_metric, self.best_epoch  # Skip saving

        tmp_save_path = os.path.join(self.checkpoint_dir, 'tmp' + self.extension)
        last_save_path = os.path.join(self.checkpoint_dir, 'last' + self.extension)
        self._save(tmp_save_path, epoch, metric)
        self._replace(tmp_save_path, last_save_path)

        worst_file = self.checkpoint_files[-1] if self.checkpoint_files else None
        if (
            len(self.checkpoint_files) < self.max_history
            or metric is None
            or self.cmp(metric, worst_file[1])
        ):
            if len(self.checkpoint_files) >= self.max_history:
                self._cleanup_checkpoints(1)
            filename = f"{self.save_prefix}-{epoch if epoch is not None else 'latest'}{self.extension}"
            save_path = os.path.join(self.checkpoint_dir, filename)
            self._duplicate(last_save_path, save_path)

            self.checkpoint_files.append((save_path, metric))
            self.checkpoint_files = sorted(
                self.checkpoint_files, key=lambda x: x[1], reverse=not self.decreasing
            )

            if metric is not None and (self.best_metric is None or self.cmp(metric, self.best_metric)):
                self.best_epoch = epoch
                self.best_metric = metric
                best_save_path = os.path.join(self.checkpoint_dir, 'model_best' + self.extension)
                self._duplicate(last_save_path, best_save_path)
        print("Checkpoint Saved")
        return (None, None) if self.best_metric is None else (self.best_metric, self.best_epoch)

    def _cleanup_checkpoints(self, trim=0):
        trim = min(len(self.checkpoint_files), trim)
        delete_index = self.max_history - trim
        if delete_index < 0 or len(self.checkpoint_files) <= delete_index:
            return
        to_delete = self.checkpoint_files[delete_index:]
        for d in to_delete:
            try:
                _logger.debug(f"Cleaning checkpoint: {d}")
                os.remove(d[0])
            except Exception as e:
                _logger.error(f"Exception '{e}' while deleting checkpoint")
        self.checkpoint_files = self.checkpoint_files[:delete_index]