
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, ModelSummary
from lightning.pytorch.loggers import TensorBoardLogger
from utils.callbacks import ConfigSnapshotCallback
import datamodules
import systems
import argparse
from utils.misc import load_config

def run(config, args):
    dm = datamodules.make(config.system.datamodule.name, config=config.system.datamodule)
    system = systems.make(config.system.name, 
                        config=config.system, 
                        load_from_checkpoint=None if not args.resume_weights_only else args.ckpt)


    logger = TensorBoardLogger(save_dir='tb_logs', name=config.name, version=config.version)
    callbacks = [ModelCheckpoint(**config.checkpoint), LearningRateMonitor(), ModelSummary(), ConfigSnapshotCallback(config)]

    trainer = pl.Trainer(devices='auto',
                        strategy='ddp', 
                        accelerator='auto',
                        logger=logger,
                        callbacks=callbacks,
                        profiler="simple",
                        **config.trainer)

    if args.pred:
        assert args.ckpt is not None
        import torch
        checkpoint = torch.load(args.ckpt)
        trainer.fit_loop.load_state_dict(checkpoint['loops']['fit_loop'])
        trainer.test_loop.load_state_dict(checkpoint['loops']['test_loop'])
        trainer.predict(system, datamodule=dm, ckpt_path=args.ckpt)
    else:
        trainer.fit(system, datamodule=dm, ckpt_path=args.ckpt if args.ckpt is not None else None)
        trainer.predict(system, datamodule=dm, ckpt_path=args.ckpt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/cage.yaml')
    parser.add_argument('--ckpt', default=None, help='path to the weights to be resumed')
    parser.add_argument(
            '--resume_weights_only',
            action='store_true',
            help='specify this argument to restore only the weights (w/o training states)'
        )
    parser.add_argument('--pred', action='store_true')


    args, extras = parser.parse_known_args()
    config = load_config(args.config, cli_args=extras)
    config.cmd_args = vars(args)

    run(config, args)
