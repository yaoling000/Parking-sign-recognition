# pipeline.py
# One-click pipeline: Step-A (YOLO detect) -> Step-B (FT+ZS OCR Remix) -> NLP rules
# Run:  python pipeline.py

import os, re, gc, json, time, textwrap
import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import torch

# ===== 固定目录（与你截图一致）=====
ROOT            = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR       = os.path.join(ROOT, "data")
YOLO_WEIGHTS    = os.path.join(ROOT, "yolo_best.pt")

STEP_A_DIR      = os.path.join(ROOT, "stepA_output")
STEP_A_JSON     = os.path.join(STEP_A_DIR, "stepA_parking_signs.json")

STEP_B_DIR      = os.path.join(ROOT, "stepB_vis3.0")
STEP_B_JSON     = os.path.join(STEP_B_DIR, "stepB_filled.json")
DET_MODEL_DIR   = os.path.join(ROOT, "inference_det_v2")  # finetuned PaddleOCR det

NLP_OUT_DIR     = os.path.join(ROOT, "nlp_output3.0")
NLP_OUT_JSON    = os.path.join(NLP_OUT_DIR, "parking_rules_nlp_final_version.json")

os.makedirs(STEP_A_DIR, exist_ok=True)
os.makedirs(STEP_B_DIR, exist_ok=True)
os.makedirs(NLP_OUT_DIR, exist_ok=True)

# ========= 公用小工具 =========
def secs(t): return f"{t:.2f}s"

def _bbox_of_poly(poly: np.ndarray):
    x1, y1 = np.min(poly[:, 0]), np.min(poly[:, 1])
    x2, y2 = np.max(poly[:, 0]), np.max(poly[:, 1])
    return float(x1), float(y1), float(x2), float(y2)

def _center_of_poly(p: np.ndarray):
    x1, y1, x2, y2 = _bbox_of_poly(p)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

def _containment_ratio(a: np.ndarray, b: np.ndarray):
    ax1, ay1, ax2, ay2 = _bbox_of_poly(a)
    bx1, by1, bx2, by2 = _bbox_of_poly(b)
    area_b = (bx2 - bx1) * (by2 - by1) + 1e-6
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    return (iw * ih) / area_b

def _iou_bbox(a: np.ndarray, b: np.ndarray):
    ax1, ay1, ax2, ay2 = _bbox_of_poly(a)
    bx1, by1, bx2, by2 = _bbox_of_poly(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union, area_a, area_b, inter

def _mk_text_panel(lines, title="INFO", width=900, pad=16, font_size=18, line_height=24):
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
        line_height = font_size + 6
    wrapped=[]
    for ln in lines:
        wrapped += textwrap.wrap(ln, width=90) if isinstance(ln, str) else [str(ln)]
    content = [f"[{title}]"] + wrapped
    h = pad*2 + line_height*len(content)
    img = Image.new("RGB", (width, h), (30, 30, 30))
    d = ImageDraw.Draw(img); y = pad
    for i, ln in enumerate(content):
        col = (255,210,0) if i==0 else (240,240,240)
        d.text((pad, y), ln, fill=col, font=font); y += line_height
    return img

def draw_boxes(pil_img, boxes, color="green", alpha=0.5, width=3):
    overlay = pil_img.copy().convert("RGB")
    d = ImageDraw.Draw(overlay)
    col_map = {"yellow": (255,210,0), "red": (255,0,0), "green": (0,200,0)}
    c = col_map.get(color, (0,200,0))
    for b in boxes:
        pts = [tuple(p) for p in b]
        d.line(pts + [pts[0]], fill=c, width=width)
    return Image.blend(pil_img, overlay, alpha)

# ========= Step-A: YOLO 检测裁剪 =========
def run_step_a():
    from ultralytics import YOLO
    from PIL import Image
    # 配置
    CONF_THR = 0.40
    IMGSZ    = 1280
    DEVICE   = 0 if torch.cuda.is_available() else "cpu"
    USE_HALF = torch.cuda.is_available()

    yolo_model = YOLO(YOLO_WEIGHTS)
    names = yolo_model.model.names
    if isinstance(names, list): names = {i:n for i,n in enumerate(names)}

    def guess_parking_sign_id(names_dict):
        for cid, name in names_dict.items():
            if "park" in str(name).lower() and "sign" in str(name).lower():
                return cid
        return 5
    PARKING_ID = guess_parking_sign_id(names)

    def safe_crop(img, x1, y1, x2, y2):
        W,H = img.size
        x1,x2 = sorted([float(x1), float(x2)])
        y1,y2 = sorted([float(y1), float(y2)])
        x1,y1 = max(0.0,x1), max(0.0,y1)
        x2,y2 = min(W, x2), min(H, y2)
        if x2-x1<2 or y2-y1<2: return None
        return img.crop((x1,y1,x2,y2))

    def iou_xyxy(a,b):
        ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
        ix1,iy1 = max(ax1,bx1), max(ay1,by1)
        ix2,iy2 = min(ax2,bx2), min(ay2,by2)
        iw,ih = max(0,ix2-ix1), max(0,iy2-iy1)
        inter = iw*ih
        area_a = max(0,ax2-ax1)*max(0,ay2-ay1)
        area_b = max(0,bx2-bx1)*max(0,by2-by1)
        union = area_a + area_b - inter + 1e-6
        return inter/union

    def center_of(b):
        x1,y1,x2,y2 = b; return (x1+x2)/2.0, (y1+y2)/2.0

    KEEP_CLASS_IDS = set(names.keys())  # 保留全部类别，由 ID 过滤
    SYMBOL_CLASS_IDS = {cid for cid in names.keys() if cid != PARKING_ID}

    images = [f for f in sorted(os.listdir(IMAGE_DIR))
              if f.lower().endswith((".jpg",".jpeg",".png"))]
    results = []

    for fn in images:
        ipath = os.path.join(IMAGE_DIR, fn)
        img   = Image.open(ipath).convert("RGB"); W,H = img.size
        res   = yolo_model.predict(ipath, conf=CONF_THR, imgsz=IMGSZ,
                                   device=DEVICE, half=USE_HALF, verbose=False)[0]
        dets=[]
        if res.boxes is not None:
            for b in res.boxes:
                cls_id = int(b.cls[0].item())
                if cls_id not in KEEP_CLASS_IDS: continue
                x1,y1,x2,y2 = map(float, b.xyxy[0].tolist())
                dets.append({
                    "cls_id": cls_id,
                    "name": names.get(cls_id, str(cls_id)),
                    "conf": float(b.conf[0].item()),
                    "xyxy": [x1,y1,x2,y2]
                })

        # 分配符号到牌
        signs = []
        for d in dets:
            if d["cls_id"] == PARKING_ID:
                signs.append({**d, "symbols":[]})
        others = [d for d in dets if d["cls_id"] != PARKING_ID and d["cls_id"] in SYMBOL_CLASS_IDS]

        for sym in others:
            best_idx, best_iou = -1, 0.0
            scx, scy = center_of(sym["xyxy"])
            for i, s in enumerate(signs):
                x1,y1,x2,y2 = s["xyxy"]
                inside = (scx>=x1 and scx<=x2 and scy>=y1 and scy<=y2)
                iou = iou_xyxy(sym["xyxy"], s["xyxy"])
                if (inside or iou>=0.50) and iou>best_iou:
                    best_idx, best_iou = i, iou
            if best_idx>=0: signs[best_idx]["symbols"].append(sym)

        img_node = {"image": fn, "width": W, "height": H, "parking_signs": []}
        for sid, s in enumerate(signs, start=1):
            x1,y1,x2,y2 = s["xyxy"]
            crop = safe_crop(img, x1,y1,x2,y2)
            if crop is None: continue
            crop_dir = os.path.join(STEP_A_DIR, "crops", "parking_signs", os.path.splitext(fn)[0])
            os.makedirs(crop_dir, exist_ok=True)
            crop_path = os.path.join(crop_dir, f"sign_{sid:02d}.jpg")
            crop.save(crop_path)

            sym_nodes=[]
            for sym in s.get("symbols", []):
                sx1,sy1,sx2,sy2 = sym["xyxy"]
                sym_nodes.append({
                    "cls_id": sym["cls_id"],
                    "name": sym["name"],
                    "conf": round(sym["conf"],4),
                    "bbox_xyxy": [round(sx1,2), round(sy1,2), round(sx2,2), round(sy2,2)]
                })

            img_node["parking_signs"].append({
                "id": sid,
                "bbox_xyxy": [round(x1,2), round(y1,2), round(x2,2), round(y2,2)],
                "crop_path": crop_path,
                "symbols": sym_nodes,
                "text": ""
            })
        results.append(img_node)

    with open(STEP_A_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total_signs = sum(len(it["parking_signs"]) for it in results)
    print(f"[Step-A] images={len(images)}, signs={total_signs}")
    return len(images), total_signs

# ========= Step-B：你的 FT+ZS Remix 版本 =========
def run_step_b():
    from paddleocr import PaddleOCR

    ZERO_DET_KW = dict(use_angle_cls=False, lang='en', det=True,  rec=False,
                       det_db_thresh=0.15, det_db_box_thresh=0.30,
                       det_db_unclip_ratio=2.2, det_limit_side_len=1280,
                       use_gpu=False)
    FT_DET_KW   = dict(use_angle_cls=False, lang='en', det=True,  rec=False,
                       det_db_thresh=0.15, det_db_box_thresh=0.30,
                       det_db_unclip_ratio=2.2, det_limit_side_len=1280,
                       use_gpu=False)

    enable_ft = False
    if os.path.isdir(DET_MODEL_DIR):
        FT_DET_KW["det_model_dir"] = DET_MODEL_DIR
        enable_ft = True
        print(f"[Step-B] Use finetuned det: {DET_MODEL_DIR}")
    else:
        print("[Step-B] Finetuned det NOT found, fallback to ZS only")

    ocr_zero_det = PaddleOCR(**ZERO_DET_KW)
    ocr_rec      = PaddleOCR(use_angle_cls=False, lang='n', det=False, rec=True, use_gpu=False) \
                   if False else PaddleOCR(use_angle_cls=False, lang='en', det=False, rec=True, use_gpu=False)
    ocr_ft_det   = PaddleOCR(**FT_DET_KW) if enable_ft else None

    SCALES = [1.0]
    FLIPS  = [False, True]
    IOU_ADD_THR, SMALL_KEEP_RATIO = 0.25, 0.40
    IOU_DUP_THR, CENTER_DIST_THR_P, CONTAIN_RATIO_THR = 0.60, 0.010, 0.85

    def dedup_single_branch(polys, img_w, img_h,
                            iou_thr=IOU_DUP_THR, center_thr_ratio=CENTER_DIST_THR_P,
                            contain_thr=CONTAIN_RATIO_THR):
        if not polys: return []
        keep, short_side = [], min(img_w, img_h)
        for p in polys:
            pcx, pcy = _center_of_poly(p); merged=False
            for i,q in enumerate(keep):
                iou, pa, qa, _ = _iou_bbox(p,q)
                qcx,qcy = _center_of_poly(q)
                cdist = ((pcx-qcx)**2 + (pcy-qcy)**2)**0.5
                contain_p_in_q = _containment_ratio(q,p)
                contain_q_in_p = _containment_ratio(p,q)
                if (iou>iou_thr) or (cdist/short_side<center_thr_ratio) \
                   or (contain_p_in_q>contain_thr) or (contain_q_in_p>contain_thr):
                    if pa<qa: keep[i]=p
                    merged=True; break
            if not merged: keep.append(p)
        return keep

    def tta_detect_and_dedup(ocr_model, pil_img, scales=SCALES, flips=FLIPS):
        if ocr_model is None: return []
        W,H = pil_img.size; all_polys=[]
        for s in scales:
            newW,newH = int(W*s), int(H*s)
            img_s = pil_img.resize((newW,newH), Image.BILINEAR)
            for flip in flips:
                img_sf = img_s.transpose(Image.FLIP_LEFT_RIGHT) if flip else img_s
                res = ocr_model.ocr(np.array(img_sf), cls=False)
                if not res or res[0] is None: continue
                polys = [np.array(line[0]) for line in res[0]]
                mapped=[]
                for p in polys:
                    q = p.copy()
                    if flip: q[:,0] = newW - q[:,0]
                    q[:,0] /= s; q[:,1] /= s
                    mapped.append(q)
                all_polys.extend(mapped)
        return dedup_single_branch(all_polys, W, H)

    def additive_fuse(ft_polys, zs_polys, img_w, img_h,
                      iou_add_thr=IOU_ADD_THR, iou_dup_thr=IOU_DUP_THR,
                      small_keep_ratio=SMALL_KEEP_RATIO,
                      center_thr_ratio=CENTER_DIST_THR_P,
                      contain_thr=CONTAIN_RATIO_THR):
        keep = list(ft_polys); short_side = min(img_w, img_h)
        for zp in zs_polys:
            max_iou, ref_area, zs_area = 0.0, None, None
            near_ref=False
            for ref in keep:
                iou, ra, za, _ = _iou_bbox(zp, ref)
                if iou>max_iou: max_iou, ref_area, zs_area = iou, ra, za
                zcx,zcy = _center_of_poly(zp); rcx,rcy = _center_of_poly(ref)
                cdist = ((zcx-rcx)**2 + (zcy-rcy)**2)**0.5
                if (cdist/short_side<center_thr_ratio) or \
                   (_containment_ratio(ref,zp)>contain_thr) or \
                   (_containment_ratio(zp,ref)>contain_thr):
                    near_ref=True
            add=False
            if (max_iou<iou_add_thr) and (not near_ref): add=True
            else:
                if zs_area is not None and ref_area is not None:
                    if (zs_area/(ref_area+1e-6) <= small_keep_ratio) and (not near_ref):
                        add=True
            if add: keep.append(zp)

        if not keep: return keep
        dedup=[]
        for p in keep:
            pcx,pcy = _center_of_poly(p); drop=False
            for i,q in enumerate(dedup):
                iou, pa, qa, _ = _iou_bbox(p,q)
                qcx,qcy = _center_of_poly(q)
                cdist = ((pcx-qcx)**2 + (pcy-qcy)**2)**0.5
                contain_p_in_q = _containment_ratio(q,p)
                contain_q_in_p = _containment_ratio(p,q)
                if (iou>IOU_DUP_THR) or (cdist/short_side<center_thr_ratio) \
                   or (contain_p_in_q>CONTAIN_RATIO_THR) or (contain_q_in_p>CONTAIN_RATIO_THR):
                    if pa>=qa: drop=True
                    else: dedup[i]=p
                    break
            if not drop: dedup.append(p)
        return dedup

    def crop_and_recognize(pil_img, boxes, rec_model):
        out=[]; W,H = pil_img.size
        for b in boxes:
            x1,y1,x2,y2 = _bbox_of_poly(b)
            x1,x2 = sorted([x1,x2]); y1,y2 = sorted([y1,y2])
            if x2<=x1 or y2<=y1: out.append((b,"")); continue
            crop = pil_img.crop((max(0,x1), max(0,y1), min(W,x2), min(H,y2)))
            res  = rec_model.ocr(np.array(crop), det=False, rec=True)
            txt  = res[0][0][0] if (res and res[0]) else ""
            out.append((b, (txt or "").upper().strip()))
        return out

    def _has_digit(s): return any(ch.isdigit() for ch in s)
    def merge_timeline_boxes(recognized_texts, y_tol=30):
        if not recognized_texts: return []
        items = sorted(recognized_texts, key=lambda x:(np.mean(x[0][:,1]), np.min(x[0][:,0])))
        lines, buf, last_y = [], [], None
        for b,t in items:
            cy = float(np.mean(b[:,1]))
            if last_y is None or abs(cy-last_y)<=y_tol:
                buf.append((b,t)); last_y=cy
            else:
                lines.append(buf); buf=[(b,t)]; last_y=cy
        if buf: lines.append(buf)
        results=[]
        for ln in lines:
            ln = sorted(ln, key=lambda x: np.min(x[0][:,0]))
            raw = "".join(t for _,t in ln).upper().replace(" ","").replace("–","-").replace("—","-")
            if _has_digit(raw):
                raw = re.sub(r'(?<!\d)(\d)(\d{2})(AM|PM)\b', r'\1:\2\3', raw)
                raw = re.sub(r'(?<!\d)(\d)(\d{2})-', r'\1:\2-', raw)
                raw = re.sub(r'-(\d)(\d{2})(AM|PM)\b', r'-\1:\2\3', raw)
                raw = re.sub(r'(\b\d{1,2})(AM|PM)\b', r'\1:00\2', raw)
                raw = re.sub(r'(\b\d{1,2})-(\d{1,2})(?=[AP]M\b)', r'\1:00-\2:00', raw)
            results.append(raw)
        return results

    # 读取 Step-A
    if not os.path.exists(STEP_A_JSON):
        raise FileNotFoundError(f"Step-A JSON not found: {STEP_A_JSON}")
    with open(STEP_A_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    data_to_process = data if isinstance(data, list) else [data]

    processed_count = 0
    for item in data_to_process:
        parking_signs = item.get("parking_signs", [])
        for ps in parking_signs:
            processed_count += 1
            crop_path = ps.get("crop_path")
            if not crop_path or not os.path.exists(crop_path):
                ps.setdefault("ocr", {}).setdefault("merged_texts", {})
                ps["ocr"]["merged_texts"].setdefault("remix","")
                ps["ocr"]["merged_texts"].setdefault("zero_shot","")
                ps.setdefault("ocr", {}).setdefault("candidates", [])
                ps["ocr"].setdefault("timeline_only", [])
                continue

            roi = Image.open(crop_path).convert("RGB"); W,H = roi.size
            zs_polys = tta_detect_and_dedup(ocr_zero_det, roi)
            ft_polys = tta_detect_and_dedup(ocr_ft_det, roi) if enable_ft else []

            fused = additive_fuse(ft_polys, zs_polys, img_w=W, img_h=H)

            rec_items_zs = crop_and_recognize(roi, zs_polys, ocr_rec)
            lines_merged_zs  = merge_timeline_boxes(rec_items_zs)
            timeline_only_zs = [t for t in lines_merged_zs if re.search(r'(AM|PM|MIDNIGHT|NOON)', t)]

            rec_items = crop_and_recognize(roi, fused, ocr_rec)
            lines_merged  = merge_timeline_boxes(rec_items)
            timeline_only = [t for t in lines_merged if re.search(r'(AM|PM|MIDNIGHT|NOON)', t)]

            merged_text_remix = " | ".join(lines_merged)    if lines_merged    else ""
            merged_text_zs    = " | ".join(lines_merged_zs) if lines_merged_zs else ""
            candidates = [c for c in {merged_text_remix, merged_text_zs} if c]

            ps.setdefault("ocr", {}).setdefault("merged_texts", {})
            ps["ocr"]["merged_texts"]["remix"]     = merged_text_remix
            ps["ocr"]["merged_texts"]["zero_shot"] = merged_text_zs
            ps["ocr"]["candidates"]    = candidates
            ps["ocr"]["timeline_only"] = list({*timeline_only, *timeline_only_zs})

            # 可视化
            base_dir  = os.path.join(STEP_B_DIR, os.path.basename(os.path.dirname(crop_path)))
            os.makedirs(base_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(crop_path))[0]
            vis_zs    = draw_boxes(roi, zs_polys, color="yellow", alpha=0.5)
            vis_ft    = draw_boxes(roi, ft_polys, color="red",    alpha=0.5)
            vis_fused = draw_boxes(roi, fused,    color="green",  alpha=0.5)
            vis_zs.save(os.path.join(base_dir, f"{base_name}__det_zs.png"))
            vis_ft.save(os.path.join(base_dir, f"{base_name}__det_ft.png"))
            vis_fused.save(os.path.join(base_dir, f"{base_name}__det_fused.png"))

            panel_remix = _mk_text_panel([f"REMIX: {merged_text_remix or '<empty>'}"], title="MERGED (REMIX)")
            panel_zs    = _mk_text_panel([f"ZERO_SHOT: {merged_text_zs or '<empty>'}"], title="MERGED (ZERO-SHOT)")
            panel_cmp   = _mk_text_panel(["Remix: " + (merged_text_remix or "<empty>"),
                                          "Zero-shot: " + (merged_text_zs or "<empty>"),
                                          "", "Candidates:"] + [f" - {c}" for c in candidates],
                                          title="COMPARE (REMIX vs ZERO-SHOT)")
            panel_remix.save(os.path.join(base_dir, f"{base_name}__remix_text.png"))
            panel_zs.save(os.path.join(base_dir, f"{base_name}__zs_text.png"))
            panel_cmp.save(os.path.join(base_dir, f"{base_name}__rec_compare.png"))
            roi.close(); gc.collect()

    with open(STEP_B_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[Step-B] processed signs = {processed_count}")
    return processed_count

# ========= NLP：解析自然语言规则 =========
# ——以下与 notebook 逻辑一致（略作封装）——
SPELL_FIX = {
    "STOPPIG":"STOPPING","STOPPIN":"STOPPING","STPPING":"STOPPING",
    "PARKIN":"PARKING","PARKlNG":"PARKING",
    "PERMITHOLDERS":"PERMIT HOLDERS","PERMITEOLDERS":"PERMIT HOLDERS",
    "PERMITHOLDER":"PERMIT HOLDERS","PERMIT_HOLDERS":"PERMIT HOLDERS",
    "PERRMIT":"PERMIT","PERM1T":"PERMIT",
    "EXCEPTEDAREA":"EXCEPTED AREA","AREAXCERSTED":"AREA EXCEPTED",
    "ZONEGG":"ZONE","Z0NE":"ZONE","PUBLICHOLIDAYYS":"PUBLIC HOLIDAYS",
    "PUBLICHOLIDAY":"PUBLIC HOLIDAYS","PUBLICHOLIDAYS":"PUBLIC HOLIDAYS",
    "PAYANC":"PAY AND","PAYAN":"PAY AND","YOURPAR":"YOUR PARK","THEPARK":"THE PARK",
    "(NOTICKE":"(NOTICE","MINMONSAT":"MIN MON-SAT","30MIN":"30 MIN","30MINUTE":"30 MIN",
    "15P6AM-9:00AM":"15P 6AM-9:00AM","MON1-FRI":"MON-FRI","MONPM":"MON","FRIPM":"FRI PM",
    "AREATSD":"AREA15","FRIAM":"FRI AM","SATPM":"SAT PM","MONSAT":"MON-SAT","SAT-SUNPM":"SAT-SUN PM",
    "ONLYP":"ONLY P","ONLY.":"ONLY","+T":"T","2RP":"2P",
}
DAY_MAP = {"MON":"Monday","TUE":"Tuesday","WED":"Wednesday","THU":"Thursday","FRI":"Friday","SAT":"Saturday","SUN":"Sunday"}

def up(s): return re.sub(r"\s+"," ", s.upper().strip())
def apply_spell_fix(s):
    t=s
    for k,v in SPELL_FIX.items(): t=t.replace(k,v)
    return re.sub(r"\s+"," ", t).strip()
def clean_token(tok):
    t=up(tok); t=t.replace(",", "").replace(";", " ").replace("—","-").replace("–","-")
    return apply_spell_fix(t)

def normalize_am_pm_noise(s):
    t=s
    t=re.sub(r"\bAM[\.\-]?PM\b","AM PM",t); t=re.sub(r"\bA\.?M\.?P\.?M\.?\b","AM PM",t)
    t=re.sub(r"\bAMP\.?M\b","AM PM",t); t=re.sub(r"\bA\.?M\b","AM",t); t=re.sub(r"\bP\.?M\b","PM",t)
    t=re.sub(r"\b(AM)(\s+AM)+\b","AM",t); t=re.sub(r"\b(PM)(\s+PM)+\b","PM",t); return t
def insert_dash_between_compact_pairs(t):
    return re.sub(r"\b(\d{1,2}(?::\d{2})?)(AM|PM)\s*(\d{1,2}(?::\d{2})?)(AM|PM)\b", r"\1\2-\3\4", t)
def expand_910_am_pm(t):
    t=re.sub(r"\b(\d{1,2})\s*[- ]\s*(\d{1,2})\s+AM\s+PM\b", r"\1AM-\2PM", t)
    t=re.sub(r"\b(\d{1,2})\s+(\d{1,2})\s+AM\s+PM\b", r"\1AM-\2PM", t)
    t=re.sub(r"\b(\d{1,2})\s*[- ]\s*(\d{1,2})\s+AMPM\b", r"\1AM-\2PM", t)
    t=re.sub(r"\b(\d{1,2})(\d{2})\s+AMPM\b", r"\1:\2 AM-\1:\2 PM", t); return t
def attach_lonely_am_pm(t):
    t=re.sub(r"\b(\d{1,2}(?::\d{2})?)\s*-\s*(\d{1,2}(?::\d{2})?)\s*PM\s*AM\b", r"\1AM-\2PM", t)
    t=re.sub(r"\b(\d{1,2}(?::\d{2})?)\s*-\s*(\d{1,2}(?::\d{2})?)\s*AM\s*PM\b", r"\1AM-\2PM", t); return t
def glue_split_time_pieces(tokens: List[str]) -> str:
    s=" ".join(tokens); s=clean_token(s); s=normalize_am_pm_noise(s)
    def _strip_zero_noise(m):
        x=m.group(0)
        if len(x)>=3 and x.count("0")/len(x)>=0.5:
            kept="".join([ch for ch in x if ch!="0"]); return " ".join(list(kept)) if kept else ""
        return x
    def _join_compact_two_times(m):
        h1,mm1,ap1 = m.group(1), m.group(2), m.group(3).upper()
        h2,mm2,ap2 = m.group(4), (m.group(5) or "00"), m.group(6).upper()
        return f"{int(h1)}:{mm1} {ap1}-{int(h2)}:{mm2} {ap2}"
    s = re.sub(r'(?i)\b(\d{1,2})(00|15|30|45)\s*(AM|PM)\s*[:\s]*([0-2]?\d)(?::?(\d{2})(?::\d{2})?)?\s*(AM|PM)\b', _join_compact_two_times, s)
    s = re.sub(r"\b(\d{1,2})\s*(AM|PM)\s*(00|15|30|45)\b", r"\1:\3 \2", s)
    s = re.sub(r"\b\d{3,}\b", _strip_zero_noise, s)
    s = insert_dash_between_compact_pairs(s); s = expand_910_am_pm(s); s = attach_lonely_am_pm(s)
    s = re.sub(r"\b(AM|PM)\s*-\s*(\d{1,2}(?::\d{2})?)\s*(AM|PM)\b", r"\1-\2\3", s)
    s = re.sub(r"\b(\d{1,2})\s*-\s*AM\s*(\d{1,2})\s*PM\b", r"\1AM-\2PM", s)
    s = re.sub(r"\b(\d{1,2})\s*-\s*PM\s*(\d{1,2})\s*AM\b", r"\1PM-\2AM", s)
    s = re.sub(r"\s*-\s*","-", s); s = re.sub(r"\s+"," ", s).strip(); return s
def extract_time_ranges(big_text: str) -> List[Tuple[str, str]]:
    text=big_text; pattern=r"\b(\d{1,2}(?::\d{2})?)\s*(AM|PM)\s*-\s*(\d{1,2}(?::\d{2})?)\s*(AM|PM)\b"
    ranges=[]
    for m in re.finditer(pattern, text):
        h1,am1,h2,am2 = m.group(1),m.group(2),m.group(3),m.group(4)
        if ":" not in h1: h1=f"{int(h1)}:00"
        if ":" not in h2: h2=f"{int(h2)}:00"
        ranges.append((f"{h1} {am1}", f"{h2} {am2}"))
    uniq=[]; [uniq.append(r) for r in ranges if r not in uniq]; return uniq
def parse_days(big_text: str) -> Optional[str]:
    t=big_text; m=re.search(r"\b(MON|TUE|WED|THU|FRI|SAT|SUN)\s*-\s*(MON|TUE|WED|THU|FRI|SAT|SUN)\b", t)
    if m: return f"{DAY_MAP[m.group(1)]} to {DAY_MAP[m.group(2)]}"
    days = list(dict.fromkeys(re.findall(r"\b(MON|TUE|WED|THU|FRI|SAT|SUN)\b", t)))
    return DAY_MAP[days[0]] if len(days)==1 else None
def parse_duration(big_text: str) -> Optional[str]:
    m=re.search(r"\b(\d{1,2})\s*P\b", big_text)
    if m:
        val=int(m.group(1));
        return f"{val}-minute limit" if val in (5,10,15,20,30,45) else f"{val}-hour limit"
    m=re.search(r"\b(\d{1,3})\s*MIN\b", big_text)
    return f"{int(m.group(1))}-minute limit" if m else None
def detect_paid(big_text: str) -> bool:
    return any(k in big_text for k in ["METER","TICKET","PAY","PAID"])
def decide_rule_type(big_text: str, paid: bool) -> str:
    if "NO STOPPING" in big_text or "NO PARKING" in big_text: return "no parking"
    return "paid parking" if paid else "free parking"
def direction_from_symbols(symbols: List[Dict]) -> Optional[str]:
    names=" ".join([s.get("name","").lower() for s in symbols])
    if "arrow_left" in names: return "on the left"
    if "arrow_right" in names: return "on the right"
    if "arrow_bidir" in names: return "in both directions"
    return None
def collect_other_info(big_text: str) -> str:
    t=big_text; infos=[]
    if "LOADING ZONE" in t: infos.append("loading zone")
    if "BUS ZONE" in t: infos.append("bus only")
    if "TAXI ZONE" in t: infos.append("taxi only")
    if "HANDICAP" in t: infos.append("accessible/disabled bay")
    if "AUTHORISED CARSHARE VEHICLES EXCEPTED" in t: infos.append("carshare vehicles excepted")
    if "PERMIT HOLDERS" in t and "EXCEPTED" in t:
        m=re.search(r"AREA\s*([A-Z]?\d+)", t)
        infos.append(f"permit holders excepted (Area {m.group(1)})" if m else "permit holders excepted")
    elif "EXCEPTED AREA" in t or "AREA" in t:
        m=re.search(r"AREA\s*([A-Z]?\d+)", t)
        if m: infos.append(f"Area {m.group(1)}")
    if "PUBLIC HOLIDAYS" in t: infos.append("including public holidays")
    infos=list(dict.fromkeys(infos)); return ", ".join(infos)
def build_sentence(rule_type, duration, days, times, direction):
    start = "No parking" if rule_type=="no parking" else ("Paid parking" if rule_type=="paid parking" else "Free parking")
    parts=[start]
    if duration: parts.append(duration)
    if isinstance(days,str) and days: parts.append(f"from {days}")
    if times:
        if isinstance(days,list) and len(days)==len(times) and len(times)>0:
            parts.append("; ".join([f"between {a} and {b}" + (f" ({d})" if d else "") for (a,b),d in zip(times,days)]))
        else:
            parts.append("; ".join([f"between {a} and {b}" for a,b in times]))
    if direction: parts.append(direction)
    s=", ".join([p for p in parts if p]).strip()
    if not s.endswith("."): s+="."
    return s

def run_nlp():
    if not os.path.exists(STEP_B_JSON):
        raise FileNotFoundError(f"Step-B JSON not found: {STEP_B_JSON}")
    with open(STEP_B_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict): raw=[raw]

    results=[]
    for img in raw:
        image_name = img.get("image","")
        for sign in img.get("parking_signs", []):
            remix = (((sign.get("ocr",{}) or {}).get("merged_texts",{}) or {}).get("remix","")).strip()
            if not remix:
                remix = (((sign.get("ocr",{}) or {}).get("merged_texts",{}) or {}).get("zero_shot","")).strip()
            if not remix: continue

            tokens = [clean_token(t) for t in remix.split("|")]; tokens=[t for t in tokens if t]
            DAY_RE = re.compile(r"^(?:MON|TUE|WED|THU|FRI|SAT|SUN)(?:\s*-\s*(?:MON|TUE|WED|THU|FRI|SAT|SUN))?$", re.I)
            time_pairs, days_list = [], []; i,N=0,len(tokens)
            while i<N:
                j=i; time_buf=[]
                while j<N and not DAY_RE.match(tokens[j]): time_buf.append(tokens[j]); j+=1
                time_text = glue_split_time_pieces(time_buf) if time_buf else ""
                pairs = extract_time_ranges(time_text); pair_first = pairs[0] if pairs else None
                k=j; day_buf=[]
                while k<N and DAY_RE.match(tokens[k]): day_buf.append(tokens[k]); k+=1
                day_text=" ".join(day_buf).upper().strip()
                def _full(dd): return (DAY_MAP.get(dd,dd)).upper()
                if day_text:
                    if "-" in day_text:
                        a,b=[x.strip() for x in day_text.split("-",1)]; day_text=f"{_full(a)}-{_full(b)}"
                    else:
                        day_text=_full(day_text)
                if pair_first:
                    time_pairs.append(pair_first); days_list.append(day_text or "")
                i = k if k>i else (i+1)

            big_text = apply_spell_fix(glue_split_time_pieces(tokens))
            time_ranges = extract_time_ranges(big_text)
            days = parse_days(big_text)
            nr_times = time_pairs if time_pairs else time_ranges
            days_for_sentence = days_list if time_pairs else days
            duration  = parse_duration(big_text)
            paid      = detect_paid(big_text)
            rule_type = decide_rule_type(big_text, paid)
            direction = direction_from_symbols(sign.get("symbols", []))
            other     = collect_other_info(big_text)

            natural = build_sentence(rule_type, duration, days_for_sentence, nr_times, direction)

            results.append({
                "image_id": image_name,
                "sign_id": sign.get("id",0),
                "natural_language": natural,
                "text": {
                    "time": nr_times,
                    "days": days_for_sentence,
                    "direction": direction,
                    "rules": rule_type,
                    "duration": duration,
                },
                "other_information": other
            })

    with open(NLP_OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[NLP] parsed signs = {len(results)}")
    return len(results)

# ========= 主入口 =========
if __name__ == "__main__":
    t0 = time.perf_counter()
    print("====== Pipeline start ======")

    ta0 = time.perf_counter()
    a_imgs, a_signs = run_step_a()
    ta1 = time.perf_counter()
    print(f"[Step-A] images={a_imgs}, signs={a_signs}, time={secs(ta1-ta0)}")
    print(f"  -> crops & JSON: {STEP_A_DIR}")

    tb0 = time.perf_counter()
    b_signs = run_step_b()
    tb1 = time.perf_counter()
    print(f"[Step-B] signs={b_signs}, time={secs(tb1-tb0)}")
    print(f"  -> vis & JSON: {STEP_B_DIR}")

    tn0 = time.perf_counter()
    n_rules = run_nlp()
    tn1 = time.perf_counter()
    print(f"[NLP] rules={n_rules}, time={secs(tn1-tn0)}")
    print(f"  -> rules JSON: {NLP_OUT_JSON}")

    t1 = time.perf_counter()
    print("====== Pipeline done ======")
    print(f"[Total] {secs(t1-t0)}")
