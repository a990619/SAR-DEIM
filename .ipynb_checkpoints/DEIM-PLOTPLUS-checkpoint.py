#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DEIM 多模型批量检测评估（强制单类 Ship，DEIM 口径速度，打印 Params/GFLOPs）
- 预处理：等距拉伸到 640x640（避免尺寸不匹配）
- 速度口径(与官方 evaluate() 一致)：
    per-image Inference(ms) = 累计前向耗时 / 图像数
    per-image Postprocess(ms) = 累计后处理耗时 / 图像数
    FPS = 1000 / (Inference + Postprocess)
- 控制台将打印：Params(M)、GFLOPs、Checkpoint 大小
- 导出：predictions_coco.json / inference_times.csv / summary.txt / model_info.txt / results_all.csv / compare_ap_fps.png
- 强制单类：COCO 输出一律 category_id=1；可视化标签固定 "Ship"
"""

import os, sys, json, csv, time, shutil, cv2
from pathlib import Path
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

# 可选：GFLOPs 统计
try:
    from thop import profile as thop_profile
except Exception:
    thop_profile = None
    print("[WARN] 未安装 thop，GFLOPs 将跳过（可 pip install thop）。")

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 你的工程依赖（按需改）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from engine.core import YAMLConfig

# ======================= 配置区 =======================
OUTPUT_ROOT          = Path("inference_results_sar")
SELECT_TAGS          = None          # 例如 {'A1','B2'}；None=全跑
SCORE_THRESHOLD      = 0.4
DEVICE_STR           = 'cuda'
INPUT_SIZE           = 640           # 等距拉伸到 640x640
USE_AMP              = False
LOW_AP_MOVE_THRES    = 0.4
IMG_EXTS             = {'.jpg', '.jpeg', '.png', '.bmp'}
WARMUP_ITERS         = 8             # 预热迭代，减轻冷启动影响
# ★★★ 按你的实际路径填写 ★★★
GROUPS = [
    {
        'input': '/public/home/userabc/SARShip/images/val',
        'annotation': '/public/home/userabc/SARShip/coco/val.json',
        'models': [
            {'config': '/public/home/userabc/DEIM_m3/configs/deim_dfine/deim_hgnetv2_n_coco.yml',
             'checkpoint': '/public/home/userabc/DEIM_m3/best_stg2.pth'},
        ]
    },
]
# =====================================================

# ====== 可视化（红框 + 红底白字 “Ship <conf>”）======
DRAW_CONF_THRESH = SCORE_THRESHOLD
BOX_COLOR       = (0, 0, 255)   # 红
BOX_THICKNESS   = 4
FONT            = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE      = 1.0
FONT_THICK      = 2
BG_PADDING      = 4

def draw_boxes_bgr(vis_bgr, xyxy, confs):
    for (x1, y1, x2, y2), conf in zip(xyxy, confs):
        if conf < DRAW_CONF_THRESH:
            continue
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(vis_bgr, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
        label = f"Ship {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICK)
        bg_tl = (x1, max(0, y1 - th - BG_PADDING))
        bg_br = (x1 + tw + BG_PADDING, y1)
        cv2.rectangle(vis_bgr, bg_tl, bg_br, BOX_COLOR, -1)
        text_org = (x1 + BG_PADDING // 2, y1 - BG_PADDING // 2)
        cv2.putText(vis_bgr, label, text_org, FONT, FONT_SCALE, (255, 255, 255), FONT_THICK, lineType=cv2.LINE_AA)
    return vis_bgr

# ====== 计时（CUDA Events）======
@torch.inference_mode()
def time_ms(device, fn):
    if torch.cuda.is_available() and device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize(device)
        return out, float(start.elapsed_time(end))
    else:
        t0 = time.perf_counter()
        out = fn()
        return out, float((time.perf_counter() - t0) * 1000.0)

GLOBAL_ROWS = []

def run_group(group, group_idx):
    coco_gt = COCO(group['annotation'])
    if 'info' not in coco_gt.dataset: coco_gt.dataset['info'] = {"description": "placeholder"}
    if 'licenses' not in coco_gt.dataset: coco_gt.dataset['licenses'] = []

    img_ids   = coco_gt.getImgIds()
    imgs_meta = coco_gt.loadImgs(img_ids)
    file2id   = {Path(x['file_name']).name.lower(): x['id'] for x in imgs_meta}

    for model_idx, m in enumerate(group['models']):
        tag = f"{chr(ord('A') + group_idx)}{model_idx + 1}"
        if SELECT_TAGS and tag not in SELECT_TAGS:
            continue

        out_root = OUTPUT_ROOT / f"{tag}_DEBUG"
        out_root.mkdir(parents=True, exist_ok=True)

        # ---- 加载 cfg & 权重 ----
        cfg = YAMLConfig(m['config'], resume=m['checkpoint'])
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False  # 评测时禁外部下载

        ckpt  = torch.load(m['checkpoint'], map_location='cpu')
        if isinstance(ckpt, dict):
            state = ckpt.get('ema', {}).get('module') or ckpt.get('model', ckpt)
        else:
            state = ckpt
        cfg.model.load_state_dict(state, strict=False)

        net  = cfg.model.deploy()
        post = cfg.postprocessor.deploy()

        class Deployed(nn.Module):
            def __init__(self, net, post):
                super().__init__()
                self.net  = net
                self.post = post
            def forward(self, imgs, sizes):
                return self.post(self.net(imgs), sizes)

        device = torch.device(DEVICE_STR if torch.cuda.is_available() and DEVICE_STR.startswith('cuda') else 'cpu')
        model  = Deployed(net, post).to(device).eval()

        # GFLOPs / Params（仅主体）
        gflops = None; param_m = None
        if thop_profile is not None:
            try:
                dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)
                macs, params = thop_profile(deepcopy(model.net), inputs=(dummy,), verbose=False)
                gflops  = macs / 1e9
                param_m = params / 1e6
            except Exception as e:
                print(f"[WARN] thop 分析失败：{e}")
        # 若 thop 失败/未装，至少给出参数量
        if param_m is None:
            try:
                param_m = sum(p.numel() for p in model.net.parameters()) / 1e6
            except Exception:
                pass

        ckpt_mb = os.path.getsize(m['checkpoint']) / (1024**2)

        # >>> 打印模型规模到控制台 <<<
        gflops_str = f"{gflops:.2f}" if gflops is not None else "N/A"
        param_str  = f"{param_m:.2f}" if param_m is not None else "N/A"
        print(f"[MODEL] {tag} — Params(M): {param_str} | GFLOPs: {gflops_str} | Checkpoint: {ckpt_mb:.2f} MB")

        # 同时单独写一个 model_info.txt
        (out_root / "model_info.txt").write_text(
            f"Params(M): {param_str}\nGFLOPs: {gflops_str}\nCheckpoint(MB): {ckpt_mb:.2f}\nInputSize: {INPUT_SIZE}x{INPUT_SIZE}\n",
            encoding="utf-8"
        )

        # 读取图片（仅取标注里存在的）
        img_dir = Path(group['input'])
        images = []
        for meta in imgs_meta:
            p = img_dir / Path(meta['file_name']).name
            if p.suffix.lower() in IMG_EXTS and p.exists():
                images.append(p)

        # 速度累计（DEIM 口径）
        total_infer_ms = 0.0
        total_post_ms  = 0.0
        n_frames       = 0

        # 明细
        times_csv_rows = []
        preds = []

        amp_ctx = torch.cuda.amp.autocast if (USE_AMP and torch.cuda.is_available() and device.type=='cuda') else torch.cpu.amp.autocast

        # ---- 预热 ----
        if WARMUP_ITERS > 0:
            dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)
            with torch.no_grad(), amp_ctx():
                for _ in range(WARMUP_ITERS):
                    _ = model.net(dummy)
                    _ = model.post(_, torch.tensor([[INPUT_SIZE, INPUT_SIZE]], device=device, dtype=torch.float32))

        for p in images:
            key = p.name.lower()
            if key not in file2id:
                continue

            img_bgr = cv2.imread(str(p))
            if img_bgr is None:
                continue
            h0, w0 = img_bgr.shape[:2]

            # 等距拉伸到 640x640
            inp_bgr = cv2.resize(img_bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
            inp_rgb = cv2.cvtColor(inp_bgr, cv2.COLOR_BGR2RGB)
            inp = torch.from_numpy(inp_rgb).permute(2,0,1).float().unsqueeze(0).to(device) / 255.0

            # 1) Inference
            with torch.no_grad(), amp_ctx():
                (raw_out), ms_infer = time_ms(device, lambda: model.net(inp))

            # 2) Postprocess（注意尺寸 WH）
            size_wh = torch.tensor([[w0, h0]], device=device, dtype=torch.float32)
            with torch.no_grad():
                (lbls, bxs, scs), ms_post = time_ms(device, lambda: model.post(raw_out, size_wh))

            total_infer_ms += ms_infer
            total_post_ms  += ms_post
            n_frames += 1

            times_csv_rows.append({'image': p.name, 'inference_ms': ms_infer, 'post_ms': ms_post, 'total_ms': ms_infer + ms_post})

            # 可视化（原图）
            bx = bxs[0].cpu().numpy()
            sc = scs[0].cpu().numpy()
            vis_bgr = img_bgr.copy()
            vis_bgr = draw_boxes_bgr(vis_bgr, bx, sc)
            cv2.imwrite(str(out_root / p.name), vis_bgr)

            # COCO 预测（强制单类）
            for b, s in zip(bx, sc):
                x1, y1, x2, y2 = map(float, b)
                preds.append({
                    'image_id': file2id[key],
                    'category_id': 1,  # Ship
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'score': float(s),
                    'segmentation': []
                })

        # 速度统计（DEIM evaluate() 口径）
        if n_frames == 0:
            speed_infer_ms = speed_post_ms = fps_deim = avg_total_ms = 0.0
        else:
            speed_infer_ms = total_infer_ms / n_frames
            speed_post_ms  = total_post_ms  / n_frames
            avg_total_ms   = speed_infer_ms + speed_post_ms
            fps_deim       = 1000.0 / avg_total_ms if avg_total_ms > 0 else 0.0

        print('-'*20, f'Test On BatchSize: 1', '-'*20)
        print('-'*20, f"Speed: {speed_infer_ms:.4f}ms inference, {speed_post_ms:.4f}ms postprocess per image", '-'*20)
        print('-'*20, f"FPS(inference+postprocess): {fps_deim:.2f}", '-'*20)

        # 写每图时间
        with (out_root / 'inference_times.csv').open('w', newline='') as f:
            wcsv = csv.DictWriter(f, fieldnames=['image', 'inference_ms', 'post_ms', 'total_ms'])
            wcsv.writeheader(); wcsv.writerows(times_csv_rows)

        # COCO 评估
        pred_path = out_root / 'predictions_coco.json'
        pred_path.write_text(json.dumps(preds, indent=2), encoding='utf-8')
        coco_dt = coco_gt.loadRes(str(pred_path))
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.imgIds = list(file2id.values())
        coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
        ap = float(coco_eval.stats[0]); ap50 = float(coco_eval.stats[1])

        # 低 AP -> 移动
        if ap < LOW_AP_MOVE_THRES:
            low_dir = OUTPUT_ROOT / "low_ap" / out_root.name
            low_dir.parent.mkdir(exist_ok=True, parents=True)
            try:
                shutil.move(str(out_root), str(low_dir))
                print(f"[WARN] {tag}_DEBUG AP={ap:.3f} < {LOW_AP_MOVE_THRES}，已移至 {low_dir}")
                out_root = low_dir
            except Exception as e:
                print(f"[WARN] 移动低 AP 目录失败：{e}")

        # summary.txt（含 Params/GFLOPs）
        with (out_root / 'summary.txt').open('w', encoding='utf-8') as f:
            f.write(f"Tag: {tag}_DEBUG\nDataset: {Path(group['input']).name}\n")
            f.write(f"Config: {Path(m['config']).name}\nCheckpoint: {Path(m['checkpoint']).name}\n")
            f.write(f"Checkpoint Size(MB): {ckpt_mb:.2f}\n")
            f.write(f"Params(M): {param_str}\n")
            f.write(f"GFLOPs: {gflops_str}\n")
            f.write(f"Inference(ms) per image (DEIM): {speed_infer_ms:.2f}\n")
            f.write(f"Postprocess(ms) per image (DEIM): {speed_post_ms:.2f}\n")
            f.write(f"Avg Total(ms) per image: {avg_total_ms:.2f}\n")
            f.write(f"FPS (inference+postprocess, DEIM-style): {fps_deim:.2f}\n\n")
            stats_name = [
                "AP@[.5:.95]", "AP@0.5", "AP@0.75", "AP@0.95",
                "AP_small", "AP_medium", "AP_large",
                "AR@1", "AR@10", "AR@100",
                "AR_small", "AR_medium", "AR_large"
            ]
            for n, v in zip(stats_name, coco_eval.stats):
                f.write(f"{n}: {float(v):.4f}\n")

        print(f"[INFO] {tag}_DEBUG 完成 — AP={ap:.3f}, AP50={ap50:.3f}, "
              f"infer={speed_infer_ms:.1f} ms, post={speed_post_ms:.1f} ms, FPS={fps_deim:.2f}")

        GLOBAL_ROWS.append({
            "tag": tag,
            "dataset": Path(group["input"]).name,
            "config": Path(m["config"]).name,
            "ckpt": Path(m["checkpoint"]).name,
            "Params(M)": float(param_m) if param_m is not None else None,
            "GFLOPs": float(gflops) if gflops is not None else None,
            "FPS": round(fps_deim, 2),
            "AvgInfer(ms)": round(speed_infer_ms, 2),
            "AvgPost(ms)": round(speed_post_ms, 2),
            "AP@[.5:.95]": round(ap, 4),
            "AP@0.5": round(ap50, 4),
        })

def finalize_and_plot():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not GLOBAL_ROWS:
        print("[WARN] 无评估结果。"); return

    # results_all.csv
    keys = list(GLOBAL_ROWS[0].keys())
    csv_path = OUTPUT_ROOT / "results_all.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(GLOBAL_ROWS)
    print(f"[OK] 汇总已写入：{csv_path.resolve()}")

    # AP@0.5 vs FPS（点大小 ~ 参数量）
    datasets = sorted(set(r["dataset"] for r in GLOBAL_ROWS))
    colors = {ds: plt.cm.tab10(i % 10) for i, ds in enumerate(datasets)}

    plt.figure(figsize=(8.5, 5.2))
    for r in GLOBAL_ROWS:
        x = float(r["FPS"]); y = float(r["AP@0.5"])
        pm = r["Params(M)"] if r["Params(M)"] is not None else 0.0
        sz = 20 + 3 * float(pm)
        c  = colors[r["dataset"]]
        plt.scatter(x, y, s=sz, color=c, alpha=0.75, edgecolors='k', linewidths=0.5)
        plt.text(x, y, r["tag"], fontsize=8, ha="left", va="bottom")

    plt.xlabel("FPS (inference+postprocess, DEIM-style) ↑")
    plt.ylabel("AP@0.5 ↑")
    plt.title("DEIM Multi-Model Benchmark (Single-Class: Ship)")
    handles = [plt.Line2D([0],[0], marker='o', color='w', label=ds,
               markerfacecolor=colors[ds], markeredgecolor='k', markersize=7) for ds in datasets]
    plt.legend(handles=handles, title="Dataset", loc="lower right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.3); plt.tight_layout()
    fig_path = OUTPUT_ROOT / "compare_ap_fps.png"
    plt.savefig(fig_path, dpi=220); plt.close()
    print(f"[OK] 对比图已写入：{fig_path.resolve()}")

if __name__ == '__main__':
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "run_settings.txt").open("w", encoding="utf-8") as f:
        f.write(f"DEVICE={DEVICE_STR}\nINPUT_SIZE={INPUT_SIZE}\nUSE_AMP={USE_AMP}\n")
        f.write(f"LOW_AP_MOVE_THRES={LOW_AP_MOVE_THRES}\nSCORE_THRESHOLD={SCORE_THRESHOLD}\n")
        f.write(f"WARMUP_ITERS={WARMUP_ITERS}\n")
        f.write("FORCE_SINGLE_CLASS=Ship (category_id=1)\n")
    if not GROUPS:
        print("[WARN] GROUPS 为空，先在脚本顶部填路径。")
    else:
        print(f"Starting inference on {len(GROUPS)} group(s)…")
        for idx, grp in enumerate(GROUPS):
            try: run_group(grp, idx)
            except Exception as e: print(f"[ERROR] 组 {idx} 评估异常：{e}")
        finalize_and_plot()
        print("All selected debug runs completed!")
