import os
import sys
import argparse
import yaml
import logging
import hashlib
import shutil
import warnings
from pathlib import Path
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from datetime import datetime
import numpy as np
import soundfile as sf
import librosa

try:
    import pyloudnorm as pyln
    HAS_PYLN = True
except ImportError:
    HAS_PYLN = False
try:
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def load_config(config_path=None, defaults=None):
    if defaults is None:
        defaults = {}
    config = defaults.copy()
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
        if yaml_config:
            config.update(yaml_config)
    return config


def get_audio_paths(input_dir, extensions):
    paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if Path(file).suffix.lower() in extensions:
                paths.append(os.path.join(root, file))
    return paths


def save_audio(y, sr, output_path, bit_depth=16):
    y = np.atleast_2d(y)
    if y.shape[0] > 2:
        y = y[:2, :]
    subtype = 'PCM_' + str(bit_depth)
    sf.write(output_path, y.T, sr, subtype=subtype)


def compute_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def deduplicate_hashes(clean_files, output_clean_dir):
    seen = set()
    kept = []
    for src in clean_files:
        md5 = compute_md5(src)
        if md5 in seen:
            os.remove(src)
            logger.info(f"Hash dup removed: {src}")
        else:
            seen.add(md5)
            kept.append(src)
    return kept


def fingerprint_dedup(clean_files, config):
    if not HAS_SKLEARN or not clean_files:
        return clean_files
    features = []
    for f in clean_files:
        y, sr = librosa.load(f, sr=config["sample_rate"], mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        features.append(mfcc.mean(axis=1))
    features = np.array(features)
    sim = cosine_similarity(features)
    kept_indices, removed_indices = [], set()
    for i in range(sim.shape[0]):
        if i in removed_indices:
            continue
        kept_indices.append(i)
        for j in range(i + 1, sim.shape[1]):
            if sim[i, j] > config["fingerprint_similarity"]:
                removed_indices.add(j)
    kept = [clean_files[i] for i in kept_indices]
    for f in (clean_files[i] for i in removed_indices):
        os.remove(f)
        logger.info(f"Fingerprint dup removed: {f}")
    return kept


def generate_metadata_csv(records, output_csv):
    import csv
    if not records:
        return
    # 收集所有 metadata 键
    keys = set()
    for rec in records:
        keys.update(rec["metadata"].keys())
    # 自定义列顺序：把最重要指标提前，其余按字母排序
    priority_keys = [
        "sample_rate", "channels", "bit_depth", "duration_sec",
        "peak_db", "rms_db", "dynamic_range_db", "loudness_lufs",
        "snr_db", "silence_ratio", "clipping_ratio",
        "source", "type"
    ]
    final_keys = []
    for k in priority_keys:
        if k in keys:
            final_keys.append(k)
            keys.discard(k)
    final_keys += sorted(keys)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["file", "status", "rejected_reason"] + final_keys)
        for rec in records:
            meta = rec.get("metadata", {})
            row = [rec["file"], rec["status"], rec.get("rejected_reason", "")]
            row += [meta.get(k, "") for k in final_keys]
            writer.writerow(row)
    logger.info(f"Metadata CSV written to {output_csv}")


def generate_report(records, output_report):
    total = len(records)
    rejected = [r for r in records if r["status"] == "rejected"]
    reasons = defaultdict(int)
    for r in rejected:
        reasons[r.get("rejected_reason", "unknown")] += 1
    lines = [
        f"===== Audio Processing Report - {datetime.now()} =====",
        f"Total files: {total}",
        f"Accepted: {total - len(rejected)}",
        f"Rejected: {len(rejected)}",
        "Rejection reasons:"
    ]
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        lines.append(f"  {reason}: {count}")
    with open(output_report, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    logger.info("\n".join(lines))


# ---------- 计算音频片段的元数据 ----------
def compute_segment_metadata(y, sr, config):
    """
    对单声道/立体声片段 y（经过 trim/标准化后的音频）计算指标。
    返回字典，键值均为可序列化的基本类型。
    """
    if y.ndim > 1:
        y_mono = np.mean(y, axis=0)
    else:
        y_mono = y

    duration = len(y_mono) / sr
    peak = np.max(np.abs(y_mono))
    peak_db = 20 * np.log10(peak) if peak > 0 else -np.inf

    rms = np.sqrt(np.mean(y_mono ** 2))
    rms_db = 20 * np.log10(rms) if rms > 0 else -np.inf

    crest = 20 * np.log10(peak / rms) if rms > 0 else 0.0

    # 响度
    loudness = None
    if HAS_PYLN:
        try:
            meter = pyln.Meter(sr)
            loudness = float(meter.integrated_loudness(y_mono if y.ndim == 1 else y))
        except:
            loudness = None

    # 静音比（基于非静音段）
    silence_samples = librosa.effects.split(y_mono, top_db=-config["silence_threshold"])
    non_silence_dur = sum((e - s) / sr for s, e in silence_samples)
    silence_ratio = 1.0 - (non_silence_dur / duration) if duration > 0 else 0.0

    # 信噪比
    if len(silence_samples) > 0:
        non_silence_mask = np.zeros_like(y_mono, dtype=bool)
        for s, e in silence_samples:
            non_silence_mask[s:e] = True
        sig_pow = np.mean(y_mono[non_silence_mask] ** 2) if np.any(non_silence_mask) else 1e-10
        noise_pow = np.mean(y_mono[~non_silence_mask] ** 2) if np.any(~non_silence_mask) else 1e-10
        snr = 10 * np.log10(sig_pow / noise_pow) if noise_pow > 0 else np.inf
    else:
        snr = np.inf

    # 削波比
    clipping_ratio = np.mean(np.abs(y_mono) >= 1.0)

    return {
        "duration_sec": round(duration, 3),
        "peak_linear": round(float(peak), 6),
        "peak_db": round(float(peak_db), 2),
        "rms_linear": round(float(rms), 6),
        "rms_db": round(float(rms_db), 2),
        "dynamic_range_db": round(float(crest), 2),
        "loudness_lufs": round(loudness, 2) if loudness is not None else None,
        "silence_ratio": round(float(silence_ratio), 4),
        "snr_db": round(float(snr), 2) if snr != np.inf else "inf",
        "clipping_ratio": round(float(clipping_ratio), 6)
    }


def process_file(args):
    file_path, config = args
    result = {
        "file": file_path,
        "status": "success",
        "errors": [],
        "metadata": {},
        "rejected_reason": None,
    }

    def reject(reason):
        result["status"] = "rejected"
        result["rejected_reason"] = reason
        return result

    # ---------- 加载 ----------
    try:
        y, sr = librosa.load(file_path, sr=None, mono=False, res_type='kaiser_best')
    except Exception as e:
        logger.error(f"无法加载 {file_path}: {e}")
        return reject("load_failed")
    if y.size == 0:
        return reject("empty_file")

    ext = Path(file_path).suffix.lower()
    if ext not in config["audio_extensions"]:
        return reject("unsupported_format")

    # ---------- 时长 ----------
    duration = len(y) / sr if y.ndim == 1 else len(y[0]) / sr
    if duration < config["min_duration"]:
        return reject("too_short")
    if duration > config["max_duration"]:
        return reject("too_long")

    # ---------- 声道处理 ----------
    target_channels = config["channels"]
    if target_channels == 1:
        if y.ndim > 1:
            y = np.mean(y, axis=0)
        y = y[np.newaxis, :] if y.ndim == 1 else y
    elif target_channels == 2 and y.ndim == 1:
        y = np.stack([y, y], axis=0)

    # ---------- 重采样 ----------
    if sr != config["sample_rate"]:
        y_resampled = []
        for ch in (y if y.ndim == 2 else [y]):
            ch_rs = librosa.resample(ch, orig_sr=sr, target_sr=config["sample_rate"], res_type='kaiser_best')
            y_resampled.append(ch_rs)
        y = np.array(y_resampled) if len(y_resampled) > 1 else y_resampled[0]
        sr = config["sample_rate"]
    y = y.squeeze()

    # ---------- 质量检测 ----------
    peak = np.max(np.abs(y))
    clipping_ratio = np.mean(np.abs(y) >= 1.0)
    if clipping_ratio > config["clip_threshold"]:
        logger.warning(f"Clipping detected: {file_path}, ratio={clipping_ratio:.4f}")
        if config.get("reject_on_clipping", False):
            return reject("clipping")

    silence_samples = librosa.effects.split(y, top_db=-config["silence_threshold"])
    non_silence_duration = sum((e - s) / sr for s, e in silence_samples)
    silence_ratio = 1.0 - (non_silence_duration / duration if duration > 0 else 0)
    if silence_ratio > config["silence_ratio_max"]:
        return reject("silence_ratio")

    if len(silence_samples) > 0:
        non_silence_mask = np.zeros_like(y, dtype=bool)
        for s, e in silence_samples:
            non_silence_mask[s:e] = True
        signal_power = np.mean(y[non_silence_mask]**2) if np.any(non_silence_mask) else 1e-10
        noise_power = np.mean(y[~non_silence_mask]**2) if np.any(~non_silence_mask) else 1e-10
        snr = 10 * np.log10(signal_power / noise_power) if noise_power != 0 else np.inf
    else:
        snr = np.inf
    if snr < config["snr_min"]:
        return reject("low_snr")

    rms = np.sqrt(np.mean(y**2))
    crest = 20 * np.log10(peak / rms) if rms > 0 else 0
    if crest < config["dynamic_range_min"]:
        return reject("low_dynamic_range")

    # ---------- 标准化前记录元数据 ----------
    result["metadata"].update({
        "sample_rate": sr,
        "channels": target_channels,
        "bit_depth": config["bit_depth"],
        "original_duration": round(duration, 3),
        "peak_db": round(20 * np.log10(peak) if peak > 0 else -np.inf, 2),
        "rms_db": round(20 * np.log10(rms) if rms > 0 else -np.inf, 2),
        "dynamic_range_db": round(crest, 2),
        "snr_db": round(float(snr), 2) if snr != np.inf else "inf",
        "silence_ratio": round(silence_ratio, 4),
        "clipping_ratio": round(clipping_ratio, 6)
    })

    if HAS_PYLN:
        try:
            meter = pyln.Meter(sr)
            result["metadata"]["loudness_lufs"] = round(float(meter.integrated_loudness(y)), 2)
        except:
            result["metadata"]["loudness_lufs"] = None

    # ---------- 标准化 ----------
    if config["peak_normalize"] and peak > 0:
        target_peak = 10 ** (config["peak_target_db"] / 20.0)
        y = y * (target_peak / peak)

    if config["lufs_normalize"] and HAS_PYLN:
        try:
            meter = pyln.Meter(sr)
            loudness = meter.integrated_loudness(y)
            y = pyln.normalize.loudness(y, loudness, config["lufs_target"])
        except Exception as e:
            logger.warning(f"LUFS normalize failed for {file_path}: {e}")

    if config["rms_normalize"]:
        current_rms = 20 * np.log10(np.sqrt(np.mean(y**2)))
        gain = 10 ** ((config["rms_target"] - current_rms) / 20.0)
        y = y * gain

    y = np.clip(y, -1.0, 1.0)

    # ---------- 静音裁剪 ----------
    if config["trim_silence"]:
        y_trimmed, _ = librosa.effects.trim(
            y,
            top_db=-config["trim_top_db"],
            frame_length=config["trim_frame_length"],
            hop_length=config["trim_hop_length"]
        )
        if len(y_trimmed) > 0:
            y = y_trimmed
            duration = len(y) / sr
            result["metadata"]["duration_after_trim"] = round(duration, 3)

    # ---------- 切片（包含每段元数据） ----------
    slices = []   # 每个元素: {"y": ndarray, "metadata": dict}
    if config["slice_enabled"] and duration > config["slice_duration"]:
        if config["slice_mode"] == "fixed":
            slice_len = int(config["slice_duration"] * sr)
            for start in range(0, len(y), slice_len):
                end = min(start + slice_len, len(y))
                seg = y[start:end]
                if len(seg) >= int(0.5 * sr):
                    seg_meta = compute_segment_metadata(seg, sr, config)
                    slices.append({"y": seg, "metadata": seg_meta})
        elif config["slice_mode"] == "silence":
            intervals = librosa.effects.split(
                y,
                top_db=-config["slice_silence_thresh"],
                frame_length=config["trim_frame_length"],
                hop_length=config["trim_hop_length"]
            )
            for s, e in intervals:
                seg = y[s:e]
                if len(seg) / sr >= 0.5:
                    seg_meta = compute_segment_metadata(seg, sr, config)
                    slices.append({"y": seg, "metadata": seg_meta})

    result["slices"] = slices
    result["y"] = y
    return result


def main():
    parser = argparse.ArgumentParser(description="批量音频DSP预处理")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args_cmd = parser.parse_args()

    config = load_config(args_cmd.config, USER_CONFIG)
    if args_cmd.input_dir:
        config["input_dir"] = args_cmd.input_dir
    if args_cmd.output_dir:
        config["output_dir"] = args_cmd.output_dir

    os.makedirs(config["output_dir"], exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(config["output_dir"], config["log_file"]))
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    logger.info("Starting DSP processing...")
    logger.info(f"Effective config: {config}")

    audio_files = get_audio_paths(config["input_dir"], config["audio_extensions"])
    logger.info(f"Found {len(audio_files)} audio files")

    pool_args = [(f, config) for f in audio_files]
    n_jobs = min(config["n_jobs"], len(audio_files)) if audio_files else 1
    results = []
    with Pool(processes=n_jobs) as pool:
        for i, res in enumerate(pool.imap_unordered(process_file, pool_args), 1):
            results.append(res)
            if i % 50 == 0:
                logger.info(f"Processed {i}/{len(audio_files)} files")

    accepted = sum(1 for r in results if r["status"] == "success")
    rejected = sum(1 for r in results if r["status"] == "rejected")
    logger.info(f"All files processed. Accepted: {accepted}, Rejected: {rejected}")

    clean_dir = os.path.join(config["output_dir"], "clean")
    rejected_dir = os.path.join(config["output_dir"], "rejected")
    slices_dir = os.path.join(config["output_dir"], "slices")
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(rejected_dir, exist_ok=True)
    if config["slice_enabled"]:
        os.makedirs(slices_dir, exist_ok=True)

    clean_file_records = []        # 所有保留的输出音频绝对路径（用于去重）
    all_metadata_records = []     # CSV元数据记录，包含主音频、切片和拒绝文件

    for res in results:
        fname = Path(res["file"]).stem
        if res["status"] == "success":
            # 保存主音频
            clean_path = os.path.join(clean_dir, fname + ".wav")
            save_audio(res["y"], config["sample_rate"], clean_path, config["bit_depth"])
            clean_file_records.append(clean_path)

            # 主音频元数据（在 process_file 中已计算好的 metadata）
            main_meta = {
                "file": clean_path,
                "status": "success",
                "rejected_reason": "",
                "metadata": {
                    "source": res["file"],
                    "type": "original",
                    **res["metadata"]      # 已包含所有关键指标
                }
            }
            all_metadata_records.append(main_meta)

            # 处理切片（每个切片自带元数据）
            for slice_info in res.get("slices", []):
                seg = slice_info["y"]
                seg_meta = slice_info["metadata"]
                idx = len(all_metadata_records)   # 简单的递增索引，也可以用实际位置
                slice_fname = f"{fname}_slice{idx:04d}"
                slice_path = os.path.join(slices_dir, f"{slice_fname}.wav")
                save_audio(seg, config["sample_rate"], slice_path, config["bit_depth"])
                clean_file_records.append(slice_path)

                slice_record = {
                    "file": slice_path,
                    "status": "success",
                    "rejected_reason": "",
                    "metadata": {
                        "source": res["file"],
                        "type": "slice",
                        "sample_rate": config["sample_rate"],
                        "channels": config["channels"],
                        "bit_depth": config["bit_depth"],
                        **seg_meta
                    }
                }
                all_metadata_records.append(slice_record)
        else:
            rejected_path = os.path.join(rejected_dir, Path(res["file"]).name)
            try:
                shutil.copy2(res["file"], rejected_path)
            except Exception as e:
                logger.error(f"Could not copy rejected file {res['file']}: {e}")

            all_metadata_records.append({
                "file": res["file"],
                "status": "rejected",
                "rejected_reason": res.get("rejected_reason", ""),
                "metadata": {
                    "type": "rejected",
                    "sample_rate": config.get("sample_rate"),
                    "channels": config.get("channels"),
                    "bit_depth": config.get("bit_depth"),
                    **res.get("metadata", {})
                }
            })

    # ---------- 去重 ----------
    if config["dedup_hash"]:
        clean_file_records = deduplicate_hashes(clean_file_records, clean_dir)
    if config["dedup_fingerprint"]:
        clean_file_records = fingerprint_dedup(clean_file_records, config)

    # 根据实际存在的文件过滤元数据（被删除的不再保留）
    existing_files = set(clean_file_records)
    final_metadata = [rec for rec in all_metadata_records
                      if rec["status"] == "rejected" or rec["file"] in existing_files]

    # ---------- 输出报告和CSV ----------
    records_for_report = [
        {"file": r["file"], "status": r["status"], "rejected_reason": r.get("rejected_reason"),
         "metadata": r.get("metadata", {})} for r in results
    ]
    generate_report(records_for_report, os.path.join(config["output_dir"], "report.txt"))

    generate_metadata_csv(final_metadata, os.path.join(config["output_dir"], "metadata.csv"))

    logger.info(f"Processing completed. Clean audio files: {len(clean_file_records)}")


USER_CONFIG = {
    # --- 基本路径 ---
    "input_dir": "F:/raw/yangqin",
    "output_dir": "E:/music-traing-data/yangqin",
    "temp_dir": "./temp",
    "log_file": "process.log",

    # --- 输出音频规格 ---
    "sample_rate": 32000,
    "channels": 1,
    "bit_depth": 16,

    # --- 输入文件过滤 ---
    "audio_extensions": [".wav", ".mp3", ".flac", ".ogg", ".m4a"],
    "min_duration": 0.5,
    "max_duration": 3000.0,

    # --- 音量标准化 ---
    "peak_normalize": True,
    "peak_target_db": -1.0,
    "lufs_normalize": True,
    "lufs_target": -23.0,
    "rms_normalize": False,
    "rms_target": -20.0,

    # --- 质量检测 ---
    "clip_threshold": 0.01,
    "reject_on_clipping": False,
    "silence_threshold": -40.0,
    "silence_ratio_max": 0.8,
    "snr_min": 10.0,
    "dynamic_range_min": 10.0,

    # --- 静音裁剪 ---
    "trim_silence": True,
    "trim_top_db": -40.0,
    "trim_frame_length": 2048,
    "trim_hop_length": 512,

    # --- 切片 ---
    "slice_enabled": True,
    "slice_mode": "fixed",
    "slice_duration": 15.0,
    "slice_silence_thresh": -40.0,

    # --- 去重 ---
    "dedup_hash": True,
    "dedup_fingerprint": False,
    "fingerprint_similarity": 0.95,

    # --- 性能 ---
    "n_jobs": cpu_count(),
    "max_retries": 3,
}

if __name__ == "__main__":
    main()