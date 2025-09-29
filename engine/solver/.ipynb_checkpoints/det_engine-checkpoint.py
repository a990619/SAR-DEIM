# engine/solver/det_engine.py
"""
DEIM: DETR with Improved Matching for Fast Convergence
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
"""
import time
import torch
from ..data import CocoEvaluator
from ..misc import MetricLogger, dist_utils
import sys, math, time, json
from typing import Iterable

import torch
import torch.amp
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils

# 终端颜色（可选）
RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"

# ==== 本地简易计时器（总时长单位：秒）====
class Profile:
    def __init__(self, device=None):
        self.is_cuda = False
        if isinstance(device, torch.device):
            self.is_cuda = (device.type == "cuda")
        elif isinstance(device, str):
            self.is_cuda = device.startswith("cuda")
        self.t = 0.0  # 累计秒
        self._start_event = None
        self._end_event = None
        self._t0 = None

    def __enter__(self):
        if self.is_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.is_cuda and torch.cuda.is_available():
            self._end_event.record()
            torch.cuda.synchronize()
            # elapsed_time 返回毫秒
            ms = self._start_event.elapsed_time(self._end_event)
            self.t += ms / 1000.0
        else:
            self.t += (time.perf_counter() - self._t0)
        return False


def train_one_epoch(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    """Train for a single epoch."""
    model.train()
    criterion.train()

    print_freq = kwargs.get('print_freq', 10)
    writer: SummaryWriter = kwargs.get('writer', None)
    ema: ModelEMA = kwargs.get('ema', None)
    scaler: GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler: Warmup = kwargs.get('lr_warmup_scheduler', None)

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    cur_iters = epoch * len(data_loader)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        if scaler is not None:
            with torch.autocast(device_type=device.type, cache_enabled=True):
                outputs = model(samples, targets=targets)

            # NaN/Inf 自检
            if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():
                print(RED + "WARNING: pred_boxes contain NaN/Inf" + RESET)
                state = {k.replace('module.', ''): v for k, v in model.state_dict().items()}
                dist_utils.save_on_master({'model': state}, "./NaN.pth")

            with torch.autocast(device_type=device.type, enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss: torch.Tensor = sum(loss_dict.values())
            scaler.scale(loss).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)
            loss: torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        if self_lr_scheduler:
            optimizer = lr_scheduler.step(cur_iters + i, optimizer)
        else:
            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

        # 日志
        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




@torch.no_grad()
def evaluate(model: torch.nn.Module,
             criterion: torch.nn.Module,
             postprocessor,
             data_loader,
             coco_evaluator: CocoEvaluator,
             device,
             test_only: bool = False,
             output_dir=None):
    """Evaluation (no grad) with optional FPS print when test_only=True."""
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    iou_types = coco_evaluator.iou_types

    # 计时累积（严格区分推理和后处理），单位：秒
    total_infer_s = 0.0
    total_post_s  = 0.0
    total_imgs    = 0

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device, non_blocking=True)
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        # --- inference ---
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = model(samples)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        # --- postprocess ---
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        t2 = time.perf_counter()
        results = postprocessor(outputs, orig_target_sizes)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t3 = time.perf_counter()

        total_infer_s += (t1 - t0)
        total_post_s  += (t3 - t2)
        total_imgs    += samples.shape[0]

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # 同步日志
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # 只在 test_only 模式下打印速度（与你之前想要的格式一致）
    if test_only and total_imgs > 0:
        infer_ms = (total_infer_s / total_imgs) * 1e3
        post_ms  = (total_post_s  / total_imgs) * 1e3
        fps = 1000.0 / (infer_ms + post_ms)
        print('-' * 20, f"Speed: {infer_ms:.4f}ms inference, {post_ms:.4f}ms postprocess per image", '-' * 20)
        print('-' * 20, f"FPS(inference+postprocess): {fps:.2f}", '-' * 20)

    # 累计 coco 指标
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    return stats, coco_evaluator
