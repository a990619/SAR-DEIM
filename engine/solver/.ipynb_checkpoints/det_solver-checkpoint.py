"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 D-FINE authors. All Rights Reserved.
"""

import time
import json
import datetime
from pathlib import Path

import torch

from ..misc import dist_utils, stats

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate
from ..optim.lr_scheduler import FlatCosineLRScheduler


class DetSolver(BaseSolver):

    # ---- 仅当 ckpt 存在时才加载 ----
    def _maybe_resume(self, ckpt_path: Path, tag: str = 'resume'):
        if ckpt_path.exists():
            print(f'>> [{tag}] Found {ckpt_path}, resume training.')
            self.load_resume_state(str(ckpt_path))
            return True
        else:
            print(f'>> [{tag}] No {ckpt_path} found, start/keep fresh.')
            return False

    def fit(self, ):
        self.train()
        args = self.cfg

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-"*42 + "Start training" + "-"*43)

        self.self_lr_scheduler = False
        if args.lrsheduler is not None:
            iter_per_epoch = len(self.train_dataloader)
            print("     ## Using Self-defined Scheduler-{} ## ".format(args.lrsheduler))
            self.lr_scheduler = FlatCosineLRScheduler(
                self.optimizer, args.lr_gamma, iter_per_epoch,
                total_epochs=args.epoches,
                warmup_iter=args.warmup_iter,
                flat_epochs=args.flat_epoch,
                no_aug_epochs=args.no_aug_epoch
            )
            self.self_lr_scheduler = True

        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        top1 = 0
        best_stat = {'epoch': -1, }

        # 评估一次（若断点续训）
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                # 训练流程下不打印 FPS
                test_only=False,
                output_dir=self.output_dir
            )
            for k in test_stats:
                best_stat['epoch'] = self.last_epoch
                best_stat[k] = test_stats[k][0]
                top1 = test_stats[k][0]
                print(f'best_stat: {best_stat}')

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            # 阶段切换点：仅当 ckpt 存在才续训
            stop_epoch = getattr(self.train_dataloader.collate_fn, 'stop_epoch', None)
            if stop_epoch is not None and stop_epoch > 0 and epoch == stop_epoch:
                ckpt = Path(self.output_dir) / 'best_stg1.pth'
                resumed = self._maybe_resume(ckpt, tag=f'resume@epoch{epoch}')
                if self.ema:
                    self.ema.decay = getattr(self.train_dataloader.collate_fn, 'ema_restart_decay', self.ema.decay)
                    print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')
                else:
                    print('>> [warn] EMA is None, skip EMA refresh.')

            train_stats = train_one_epoch(
                self.self_lr_scheduler,
                self.lr_scheduler,
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer
            )

            if not self.self_lr_scheduler:
                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                    self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir and (stop_epoch is None or epoch < stop_epoch):
                checkpoint_paths = [self.output_dir / 'last.pth']
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                # 训练流程下不打印 FPS
                test_only=False,
                output_dir=self.output_dir
            )

            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)

                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat[k] > top1:
                    best_stat_print['epoch'] = epoch
                    top1 = best_stat[k]
                    if self.output_dir:
                        if stop_epoch is not None and epoch >= stop_epoch:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg2.pth')
                        else:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg1.pth')

                best_stat_print[k] = max(best_stat[k], top1)
                print(f'best_stat: {best_stat_print}')

                if best_stat['epoch'] == epoch and self.output_dir:
                    if stop_epoch is not None and epoch >= stop_epoch:
                        if test_stats[k][0] > top1:
                            top1 = test_stats[k][0]
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg2.pth')
                    else:
                        top1 = max(test_stats[k][0], top1)
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg1.pth')

                elif stop_epoch is not None and epoch >= stop_epoch:
                    best_stat = {'epoch': -1, }
                    if self.ema:
                        self.ema.decay = max(0.0, self.ema.decay - 0.0001)
                    else:
                        print('>> [warn] EMA is None, skip EMA decay tuning.')

                    ckpt = Path(self.output_dir) / 'best_stg1.pth'
                    self._maybe_resume(ckpt, tag=f'refresh@epoch{epoch}')
                    if self.ema:
                        print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                       self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
    def val(self):
        self.eval()

        # 确保输出目录存在，避免 RuntimeError: Parent directory ... does not exist
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        module = self.ema.module if self.ema else self.model

        # 关键：test_only=True 才会在 evaluate() 里打印 FPS
        test_stats, coco_evaluator = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            self.evaluator,
            self.device,
            test_only=True,
            output_dir=self.output_dir
        )

        # 可选：保存一份 COCO eval 的状态
        if self.output_dir and coco_evaluator is not None and "bbox" in coco_evaluator.coco_eval:
            from ..misc import dist_utils
            dist_utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval,
                self.output_dir / "eval.pth"
            )
        return


