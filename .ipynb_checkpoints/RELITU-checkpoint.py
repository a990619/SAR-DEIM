#!/usr/bin/env python3
# coding: utf-8
import warnings
warnings.filterwarnings('ignore'); warnings.simplefilter('ignore')

import os, shutil, cv2, torch, gc, hashlib
import numpy as np
from PIL import Image
from tqdm import trange
import torchvision.transforms as T

from pytorch_grad_cam import (
    GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM
)
from pytorch_grad_cam.utils.image import scale_cam_image

# === 你的工程 ===
from engine.core import YAMLConfig
from tools.inference.torch_inf import draw

# -------- CLI colors ----------
RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
CLASS_NAME = None

# ================== 稳定&省显存设置（保持一致） ==================
USE_AMP = True                 # 半精度 autocast
IMG_SIDE = 640                 # 保持与位置编码一致（避免 256 vs 400 的维度冲突）

# 关键：把每个模式最多取的层数调成总和=50（A多→L少，全部开启）——保持你的逻辑
PER_MODE_CAP = {
    'A': 7, 'B': 6, 'C': 6, 'D': 5,
    'E': 5, 'F': 5, 'G': 4, 'H': 4,
    'I': 3, 'J': 2, 'K': 2, 'L': 1
}
MODES_TO_RUN = "ABCDEFGHIJKL"  # 默认 12 套都开；需要少跑可改 "ABCD" 等

# —— 新增：去重与“蓝图”判定的开关/阈值（其余不改你的流程）——
DEDUP_CAM = True               # True: 对重复 CAM 去重跳过
CAM_STD_EPS = 1e-4             # CAM 方差阈值：过小视为“全蓝/无效”
CAM_UNIQUE_ROUND = 3           # CAM 去重时小数位四舍五入
REPORT_EVERY = 10              # 每 N 张图汇报一次“跳过统计”

def _print_total_caps():
    total = sum(PER_MODE_CAP.get(k, 0) for k in MODES_TO_RUN)
    print(f"{GREEN}[Info]{RESET} 计划每张图输出最多 {total} 张（总和来自 PER_MODE_CAP + MODES_TO_RUN；遇到无效/重复将跳过）。")
_print_total_caps()

# ----------------------------------------------------------------------
class ActivationsAndGradients:
    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for tl in target_layers:
            self.handles.append(tl.register_forward_hook(self.save_activation))
            self.handles.append(tl.register_forward_hook(self.save_gradient))

    def save_activation(self, module, _input, output):
        act = output
        if self.reshape_transform is not None:
            act = self.reshape_transform(act)
        self.activations.append(act.detach().cpu())

    def save_gradient(self, module, _input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            return
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.detach().cpu()] + self.gradients
        output.register_hook(_store_grad)

    def post_process(self, result):
        boxes, logits = result['pred_boxes'], result['pred_logits']
        sorted_scores, indices = torch.sort(logits.max(2)[0], descending=True)
        return logits[0, indices][0], boxes[0, indices][0]

    def __call__(self, x):
        self.gradients.clear(); self.activations.clear()
        out = self.model(x)
        logits, boxes = self.post_process(out)
        return [[logits, boxes]]

    def release(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.gradients.clear()
        self.activations.clear()

# ----------------------------------------------------------------------
class deim_target(torch.nn.Module):
    def __init__(self, ouput_type, conf, ratio) -> None:
        super().__init__()
        self.ouput_type = ouput_type
        self.conf = conf
        self.ratio = ratio
    
    def forward(self, data):
        logits, boxes = data
        result = []
        count = int(max(1, logits.size(0) * self.ratio))
        valid_any = False
        for i in trange(count, leave=False, disable=True):
            mx = float(logits[i].max())
            if mx < self.conf:
                continue
            valid_any = True
            if self.ouput_type in ('class', 'all'):
                result.append(logits[i].max())
            if self.ouput_type in ('box', 'all'):
                for j in range(4):
                    result.append(boxes[i, j])
        # 保持你的行为：若没有符合阈值的则返回一个 0 梯度目标
        return sum(result) if result else (logits[:1].max() * 0.0)

# --------------------- CAM 形状增强（保持一致） -------------------------
def shape_cam(cam01, gamma=1.0, percentile=(0.0, 100.0),
              multiplier=1.0, dilate_ksize=0, blur_ksize=0):
    cam = cam01.astype(np.float32)
    cam = np.maximum(cam, 0)
    p_low, p_high = percentile
    if p_high <= p_low: p_low, p_high = 0.0, 100.0
    lo = np.percentile(cam, p_low); hi = np.percentile(cam, p_high)
    if hi > lo:
        cam = (cam - lo) / max(1e-6, (hi - lo))
        cam = np.clip(cam, 0, 1)
    if gamma is not None and gamma > 0: cam = np.power(cam, gamma)
    if multiplier is not None and multiplier != 1.0: cam = np.clip(cam * float(multiplier), 0, 1)
    if dilate_ksize and int(dilate_ksize) > 0:
        k = int(dilate_ksize);  k += (k % 2 == 0)
        kernel = np.ones((k, k), np.uint8)
        cam = cv2.dilate(cam, kernel, iterations=1); cam = np.clip(cam, 0, 1)
    if blur_ksize and int(blur_ksize) > 0:
        k = int(blur_ksize);  k += (k % 2 == 0)
        cam = cv2.GaussianBlur(cam, (k, k), 0); cam = np.clip(cam, 0, 1)
    return cam

def overlay_cam_on_image(img_float01, cam01, alpha=0.6, colormap=cv2.COLORMAP_JET):
    cam_uint8 = np.uint8(255 * np.clip(cam01, 0, 1))
    heatmap = cv2.applyColorMap(cam_uint8, colormap)[:, :, ::-1]
    base = np.uint8(np.clip(img_float01, 0, 1) * 255)
    out = cv2.addWeighted(base, 1 - alpha, heatmap, alpha, 0)
    return out

# ----------------------------------------------------------------------
# 选层逻辑：仅改变“选哪些层”，其它流程不动
def list_all_conv_modules(model):
    convs = {}
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            convs[name] = m
    return convs

def conv_heuristic_score(name: str) -> float:
    s = 0.0
    if ".stages.3" in name: s += 3.0
    if ".stages.2" in name: s += 2.0
    if ".stages.1" in name: s += 1.0
    if "encoder.input_proj" in name: s += 2.0
    if "encoder.lateral_convs" in name: s += 1.5
    if "encoder.fpn_blocks" in name: s += 2.0
    if "encoder.pan_blocks" in name: s += 2.0
    if "fusion_model" in name: s += 2.5
    if "feature_conv" in name: s += 2.5
    if name.endswith(".conv"): s += 0.5
    if ".cv3.1" in name or ".cv4" in name: s += 0.5
    if "downsample" in name: s += 0.5
    return s

def pick_layers_deterministic(sorted_names, k, offset, stride):
    k = max(1, min(k, len(sorted_names)))
    out, idx = [], offset % len(sorted_names)
    while len(out) < k:
        out.append(sorted_names[idx % len(sorted_names)])
        idx += max(1, stride)
    return out

def cam_hash(gray_small):
    # 归一化到 0..1，四舍五入降低噪声，然后 sha1
    g = gray_small.astype(np.float32)
    if g.size == 0:
        return None
    mi, ma = float(g.min()), float(g.max())
    if ma > mi:
        g = (g - mi) / (ma - mi)
    g = np.round(g, CAM_UNIQUE_ROUND)
    return hashlib.sha1(g.tobytes()).hexdigest()

# ----------------------------------------------------------------------
class deim_heatmap:
    def __init__(self, config, weight, device,
                 method="LayerCAM",
                 backward_type="class", conf_threshold=0.3, ratio=0.3,
                 show_box=False, renormalize=False, isUltralytics=False):
        self.device = torch.device(device)
        model, postprocessor = self.init_model(config, weight)
        self.model = model.to(self.device).eval()
        self.postprocessor_fn = postprocessor
        self.target = deim_target(backward_type, conf_threshold, ratio)
        self.method = method
        self.show_box = show_box
        self.renormalize = renormalize
        self.isUltralytics = isUltralytics
        self.conf_threshold = conf_threshold

        # 预处理（保持 640）
        self.tfm = T.Compose([T.Resize((IMG_SIDE, IMG_SIDE)), T.ToTensor()])

    def init_model(self, config, weight):
        cfg = YAMLConfig(config, resume=weight)
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False
        ckpt = torch.load(weight, map_location='cpu')
        global CLASS_NAME
        if isinstance(ckpt, dict) and ckpt.get('name', None) is not None:
            CLASS_NAME = ckpt['name']
        if isinstance(ckpt, dict):
            state = ckpt.get('ema', {}).get('module', None) or ckpt.get('model', ckpt)
        else:
            state = ckpt
        cfg.model.load_state_dict(state, strict=False)
        return cfg.model, cfg.postprocessor.deploy()

    def forward_once(self, im_data):
        with torch.no_grad():
            if USE_AMP:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    return self.model(im_data)
            else:
                return self.model(im_data)

    def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam, alpha=0.6):
        h, w, _ = image_float_np.shape
        renorm = np.zeros_like(grayscale_cam, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(int(x1), 0), max(int(y1), 0)
            x2, y2 = min(int(x2), w - 1), min(int(y2), h - 1)
            if x2 > x1 and y2 > y1:
                renorm[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        renorm = scale_cam_image(renorm)
        return overlay_cam_on_image(image_float_np, renorm, alpha=alpha)

    def make_cam(self, im_data, layer_module, cam_params):
        method_cls = eval(self.method)
        cam = method_cls(self.model, [layer_module])
        cam.activations_and_grads = ActivationsAndGradients(self.model, [layer_module], None)
        try:
            if USE_AMP:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    cam_raw = cam(im_data, [self.target])
            else:
                cam_raw = cam(im_data, [self.target])
        except RuntimeError as e:
            cam.activations_and_grads.release(); del cam
            if 'CUDA out of memory' in str(e):
                torch.cuda.empty_cache(); gc.collect()
                raise RuntimeError("OOM_skip_this_layer")
            raise
        if cam_raw is None:
            cam.activations_and_grads.release(); del cam
            raise RuntimeError("CAM returned None.")
        grayscale_cam = cam_raw[0, :]
        cam.activations_and_grads.release(); del cam

        # —— 新增：蓝图/无效判定 ——（其余不动）
        if not np.isfinite(grayscale_cam).all():
            raise RuntimeError("CAM_invalid_value")
        stdv = float(np.std(grayscale_cam))
        if stdv < CAM_STD_EPS:
            raise RuntimeError("CAM_flat")  # 将被上层捕获并告警/跳过

        # 归一化到 0..1
        grayscale_cam = np.maximum(grayscale_cam, 0)
        mi, ma = float(grayscale_cam.min()), float(grayscale_cam.max())
        if ma > mi:
            grayscale_cam = (grayscale_cam - mi) / (ma - mi)
        return grayscale_cam

    def process_one_image(self, img_path, out_dir, conv_map, mode_presets, mode_layer_names):
        im_pil = Image.open(img_path).convert('RGB')
        w, h = im_pil.size
        base = np.array(im_pil).astype(np.float32) / 255.0
        im_data = self.tfm(im_pil).unsqueeze(0).to(self.device)

        # ---- 前向 + 位置编码不匹配自动回退兜底（保持一致） ----
        try:
            pred = self.forward_once(im_data)
        except RuntimeError as e:
            msg = str(e)
            if ("The size of tensor a" in msg and "must match the size of tensor b" in msg) or ("pos_embed" in msg):
                print(YELLOW + "[Auto-fallback] pos_embed 尺寸不匹配，改用 640 输入重新前向。" + RESET)
                self.tfm = T.Compose([T.Resize((640, 640)), T.ToTensor()])
                im_data = self.tfm(im_pil).unsqueeze(0).to(self.device)
                pred = self.forward_once(im_data)
            else:
                raise
        orig_size = torch.tensor([[w, h]], device=self.device)

        # —— 新增：检测框情况提示（不改变你的阈值逻辑）——
        try:
            labels, boxes, scores = self.postprocessor_fn(pred, orig_size)
            num_keep = int((scores > self.conf_threshold).sum().item())
            if num_keep == 0:
                print(YELLOW + f"[Warn] {os.path.basename(img_path)} 当前 conf={self.conf_threshold} 没有检测框；Grad-CAM 可能偏平/发蓝。" + RESET)
        except Exception:
            pass

        total_count, skip_flat, skip_dup = 0, 0, 0
        seen_hashes = set()

        for mode_key, cam_params in mode_presets.items():
            if mode_key not in MODES_TO_RUN:
                continue
            names = mode_layer_names[mode_key][:PER_MODE_CAP.get(mode_key, 0)]
            for idx, name in enumerate(names, start=1):
                try:
                    gray_small = self.make_cam(im_data, conv_map[name], cam_params)

                    # —— 去重：同一图片多层若 CAM 一样则跳过 —— 
                    if DEDUP_CAM:
                        hcode = cam_hash(gray_small)
                        if hcode is not None and hcode in seen_hashes:
                            skip_dup += 1
                            continue
                        if hcode is not None:
                            seen_hashes.add(hcode)

                    gray = cv2.resize(gray_small, (w, h), interpolation=cv2.INTER_LINEAR)
                    gray = shape_cam(
                        gray,
                        gamma=cam_params['cam_gamma'],
                        percentile=cam_params['cam_percentile'],
                        multiplier=cam_params['cam_multiplier'],
                        dilate_ksize=cam_params['cam_dilate'],
                        blur_ksize=cam_params['cam_blur']
                    )
                    if self.renormalize:
                        labels, boxes, scores = self.postprocessor_fn(pred, orig_size)
                        boxes_keep = boxes[scores > self.conf_threshold].detach().cpu().numpy().astype(np.int32)
                        out_img = self.renormalize_cam_in_bounding_boxes(
                            boxes_keep, base, gray, alpha=cam_params['alpha'])
                    else:
                        # —— 保持原始方式：叠在原图上 —— 
                        out_img = overlay_cam_on_image(base, gray, alpha=cam_params['alpha'])

                    if self.show_box:
                        labels, boxes, scores = self.postprocessor_fn(pred, orig_size)
                        out_pil = Image.fromarray(out_img)
                        draw([out_pil], labels, boxes, scores,
                             thrh=self.conf_threshold, font_size_factor=0.05,
                             box_thickness_factor=0.005, class_name=CLASS_NAME)
                        out_img = np.array(out_pil)

                    stem, ext = os.path.splitext(os.path.basename(img_path))
                    ext = ext if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp'] else '.png'
                    out_path = os.path.join(out_dir, f"{stem}_{mode_key}{idx}{ext}")
                    Image.fromarray(out_img).save(out_path)
                    total_count += 1

                except RuntimeError as e:
                    if str(e) == "OOM_skip_this_layer":
                        print(YELLOW + f"[Skip OOM] {os.path.basename(img_path)} | layer:{name}" + RESET)
                    elif str(e) == "CAM_flat":
                        skip_flat += 1
                        # 仅告警不保存“全蓝/几乎全蓝”的图
                        if skip_flat % 5 == 1:  # 避免刷屏
                            print(YELLOW + f"[Warn] {os.path.basename(img_path)} | {mode_key}-{name} CAM 近乎常数，已跳过。" + RESET)
                    elif str(e) == "CAM_invalid_value":
                        print(RED + f"[Error] {os.path.basename(img_path)} | {mode_key}-{name} CAM含NaN/Inf，跳过。" + RESET)
                    else:
                        print(RED + f"[GradCAM Error] {os.path.basename(img_path)} | {mode_key}-{name}: {e}" + RESET)
                finally:
                    torch.cuda.empty_cache(); gc.collect()

        # 汇报该图统计
        if skip_flat or skip_dup:
            print(ORANGE + f"[Note] {os.path.basename(img_path)} 跳过 flat:{skip_flat} 重复:{skip_dup}，有效输出:{total_count}。" + RESET)
        else:
            print(f"{GREEN}[Info]{RESET} {os.path.basename(img_path)} 共输出 {total_count} 张。")

        im_pil.close()
        del im_data, pred
        torch.cuda.empty_cache(); gc.collect()

    def __call__(self, img_path, out_dir, mode_presets, mode_layer_names):
        if os.path.exists(out_dir): shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        conv_map = list_all_conv_modules(self.model)
        if len(conv_map) == 0:
            print(RED + "[Error] 未发现任何 Conv2d 层。" + RESET); return

        if os.path.isdir(img_path):
            files = [f for f in os.listdir(img_path)
                     if os.path.splitext(f)[-1].lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
            files.sort()
            for i, fn in enumerate(files, start=1):
                self.process_one_image(os.path.join(img_path, fn), out_dir, conv_map, mode_presets, mode_layer_names)
                if i % REPORT_EVERY == 0:
                    print(BLUE + f"[Prog] 已处理 {i}/{len(files)} 张。" + RESET)
        else:
            self.process_one_image(img_path, out_dir, conv_map, mode_presets, mode_layer_names)
        print(GREEN + f"[Done] Saved to: {out_dir}" + RESET)

# ----------------------------------------------------------------------
# 仅改变“选层集合”的策略，其它设置不变
def build_modes_and_layers_12(model):
    # 12 套“克制偏亮”的参数（保持与之前一致）
    mode_presets = {
        'A': dict(cam_gamma=1.45, cam_percentile=(3,  99.80), cam_multiplier=1.08, cam_dilate=0, cam_blur=0, alpha=0.60),
        'B': dict(cam_gamma=1.50, cam_percentile=(4,  99.85), cam_multiplier=1.06, cam_dilate=0, cam_blur=0, alpha=0.59),
        'C': dict(cam_gamma=1.55, cam_percentile=(5,  99.85), cam_multiplier=1.04, cam_dilate=0, cam_blur=0, alpha=0.58),
        'D': dict(cam_gamma=1.60, cam_percentile=(5,  99.90), cam_multiplier=1.02, cam_dilate=0, cam_blur=0, alpha=0.57),
        'E': dict(cam_gamma=1.65, cam_percentile=(6,  99.90), cam_multiplier=1.00, cam_dilate=0, cam_blur=2, alpha=0.56),
        'F': dict(cam_gamma=1.70, cam_percentile=(6,  99.90), cam_multiplier=0.98, cam_dilate=0, cam_blur=2, alpha=0.55),
        'G': dict(cam_gamma=1.75, cam_percentile=(7,  99.90), cam_multiplier=0.97, cam_dilate=0, cam_blur=3, alpha=0.54),
        'H': dict(cam_gamma=1.80, cam_percentile=(7,  99.92), cam_multiplier=0.96, cam_dilate=0, cam_blur=3, alpha=0.53),
        'I': dict(cam_gamma=1.85, cam_percentile=(8,  99.92), cam_multiplier=0.95, cam_dilate=0, cam_blur=3, alpha=0.52),
        'J': dict(cam_gamma=1.90, cam_percentile=(8,  99.95), cam_multiplier=0.94, cam_dilate=0, cam_blur=3, alpha=0.51),
        'K': dict(cam_gamma=1.95, cam_percentile=(9,  99.95), cam_multiplier=0.93, cam_dilate=0, cam_blur=3, alpha=0.50),
        'L': dict(cam_gamma=2.00, cam_percentile=(10, 99.95), cam_multiplier=0.92, cam_dilate=0, cam_blur=3, alpha=0.50),
    }

    # —— 选层：按启发式打分，然后用固定步长/偏移挑选，保证可复现 ——（只改变“选层”，不改变其它）
    convs = list_all_conv_modules(model)
    names = list(convs.keys())
    names_sorted = sorted(names, key=lambda n: (-conv_heuristic_score(n), n))

    total = len(names_sorted)
    counts = {
        'A': min(max(12, total // 6 + 10), 24), 'B': min(max(11, total // 7 + 9), 22),
        'C': min(max(10, total // 8 + 8), 20),  'D': min(max( 9, total // 9 + 7), 18),
        'E': min(max( 8, total //10 + 6), 16),  'F': min(max( 7, total //11 + 6), 14),
        'G': min(max( 6, total //12 + 5), 12),  'H': min(max( 6, total //13 + 5), 12),
        'I': min(max( 5, total //14 + 4), 10),  'J': min(max( 4, total //16 + 3),  8),
        'K': min(max( 3, total //18 + 2),  6),  'L': min(max( 2, total //20 + 2),  5),
    }

    strides = [3, 4, 5]  # A-D 更密、E-H 次之、I-L 最稀
    mode_layer_names = {}
    keys = list("ABCDEFGHIJKL")
    for idx, key in enumerate(keys):
        group = idx // 4
        stride = strides[group]
        offset = idx % 4
        k = counts[key]
        mode_layer_names[key] = pick_layers_deterministic(names_sorted, k, offset, stride)

    return mode_presets, mode_layer_names

# ----------------------------------------------------------------------
def get_params():
    # —— 按你最新给的路径 —— 
    return {
        'config': '/public/home/userabc/DEIM_m3/configs/deim_dfine/SSDD1.yml',
        'weight': '/public/home/userabc/DEIM_m3/deim_outputs/IDFINAL/best_stg2.pth',
        'device': 'cuda:0',
        'method': 'LayerCAM',
        'backward_type': 'class',
        'conf_threshold': 0.3,
        'ratio': 0.3,
        'show_box': False,
        'renormalize': False,   # 按原始：False 时叠在原图；True 时仅框内归一化后叠原图
        'isUltralytics': False,
    }

# ----------------------------------------------------------------------
if __name__ == '__main__':
    params = get_params()
    deim = deim_heatmap(**params)

    mode_presets, mode_layer_names = build_modes_and_layers_12(deim.model)
    IMG_DIR = '/public/home/userabc/SAR-PIC/ID-原图'
    OUT_DIR = '/public/home/userabc/DEIMGAI-ID_12modes_50each'

    deim(IMG_DIR, OUT_DIR, mode_presets, mode_layer_names)
