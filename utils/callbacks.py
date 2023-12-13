import shutil
import os
from utils.misc import dump_config
from lightning.pytorch.callbacks.callback import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
class ConfigSnapshotCallback(Callback):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, trainer, pl_module, stage) -> None:
        self.savedir = os.path.join(pl_module.logger.log_dir, 'config')
    
    @rank_zero_only
    def save_config_snapshot(self):
        os.makedirs(self.savedir, exist_ok=True)
        dump_config(os.path.join(self.savedir, 'parsed.yaml'), self.config)
        shutil.copyfile(self.config.cmd_args['config'], os.path.join(self.savedir, 'raw.yaml'))

    def on_fit_start(self, trainer, pl_module):
        self.save_config_snapshot()
