import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import librosa
import numpy as np

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AudioAnalyzer:
    """音频特征分析器（调式、速度）"""

    @classmethod
    def analyze_bpm(cls, y: np.ndarray, sr: int) -> Optional[str]:
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            bpm = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
            if bpm > 120:
                half_bpm = bpm / 2
                if 40 <= half_bpm <= 66 or 86 <= half_bpm <= 120:
                    bpm = half_bpm
            if 40 <= bpm <= 66:
                return "Slow"
            elif 86 <= bpm <= 120:
                return "Medium"
            elif 120 < bpm <= 168:
                return "Fast"
            else:
                return None
        except Exception as e:
            logger.warning(f"BPM analysis error: {e}")
            return None

    @classmethod
    def analyze_mode(cls, y: np.ndarray, sr: int) -> Optional[str]:
        try:
            f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
                             sr=sr, frame_length=1024, hop_length=512)
            if f0 is None or np.all(np.isnan(f0)):
                return None
            voiced_flag = ~np.isnan(f0)
            valid_idx = np.where(voiced_flag)[0]
            if len(valid_idx) < 30:
                return None
            midi_pitch = librosa.hz_to_midi(f0[valid_idx])
            pitch_class = np.round(midi_pitch).astype(int) % 12
            frame_duration = 512 / sr
            unique_classes = set(pitch_class)
            class_duration = {pc: np.sum(pitch_class == pc) * frame_duration for pc in unique_classes}
            if len(unique_classes) < 3:
                return None

            candidate_gongs = [pc for pc in unique_classes if (pc + 4) % 12 in unique_classes]
            gong = None
            if len(candidate_gongs) == 1:
                gong = candidate_gongs[0]
            elif len(candidate_gongs) > 1:
                valid = [c for c in candidate_gongs
                         if unique_classes.issubset({c, (c + 2) % 12, (c + 4) % 12, (c + 7) % 12, (c + 9) % 12})]
                if len(valid) == 1:
                    gong = valid[0]
                else:
                    pool = valid if valid else candidate_gongs
                    gong = max(pool, key=lambda x: class_duration.get(x, 0))
            else:
                valid = [c for c in unique_classes
                         if unique_classes.issubset({c, (c + 2) % 12, (c + 4) % 12, (c + 7) % 12, (c + 9) % 12})]
                if valid:
                    gong = max(valid, key=lambda x: class_duration.get(x, 0))
                else:
                    return None
            if gong is None:
                return None

            def relative_to_changming(pc):
                d = (pc - gong) % 12
                mapping = {0: 'do', 2: 're', 4: 'mi', 5: 'fa', 6: '#fa', 7: 'sol', 9: 'la', 10: 'bsi', 11: 'si'}
                return mapping.get(d)

            zhengyin_set = {'do', 're', 'mi', 'sol', 'la'}
            b_set = {relative_to_changming(pc) for pc in unique_classes
                     if relative_to_changming(pc) and relative_to_changming(pc) not in zhengyin_set}
            if len(b_set) == 0:
                mode_type = "五声"
            elif b_set == {'fa', 'si'}:
                mode_type = "清乐七声"
            elif b_set == {'#fa', 'si'}:
                mode_type = "雅乐七声"
            elif b_set == {'fa', 'bsi'}:
                mode_type = "燕乐七声"
            else:
                mode_type = "五声"

            sorted_classes = sorted(unique_classes, key=lambda x: class_duration.get(x, 0), reverse=True)
            candidate_tonic = sorted_classes[0]
            cm_tonic = relative_to_changming(candidate_tonic)
            if cm_tonic in zhengyin_set:
                tonic = candidate_tonic
            else:
                zhengyin_classes = [pc for pc in unique_classes if relative_to_changming(pc) in zhengyin_set]
                if not zhengyin_classes:
                    return None
                tonic = max(zhengyin_classes, key=lambda x: class_duration.get(x, 0))

            d = (tonic - gong) % 12
            jie_mapping = {0: 'GONG', 2: 'SHANG', 4: 'JIAO', 7: 'ZHI', 9: 'YU'}
            jie_name = jie_mapping.get(d)
            if jie_name is None:
                return None
            pitch_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            tonic_name = pitch_names[tonic]

            if mode_type == "五声":
                return f"{tonic_name} {jie_name} WUSHENG"
            else:
                type_en = {"清乐七声": "QINGYUE", "雅乐七声": "YAYUE", "燕乐七声": "YANYUE"}
                return f"{tonic_name} {jie_name} {type_en[mode_type]}"
        except Exception as e:
            logger.warning(f"Mode analysis error: {e}")
            return None


def analyze_file(audio_path: Path, target_sr: int = 32000) -> dict:
    """分析单个音频文件，返回字典"""
    song_id = audio_path.stem
    file_path = str(audio_path.resolve())
    result = {
        "song_id": song_id,
        "file_path": file_path,
        "title": song_id,
        "features": {
            "bpm": None,
            "key": None
        }
    }
    try:
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        result["features"]["bpm"] = AudioAnalyzer.analyze_bpm(y, sr)
        result["features"]["key"] = AudioAnalyzer.analyze_mode(y, sr)
    except Exception as e:
        logger.error(f"Failed to process {audio_path}: {e}")
    return result


# ╔══════════════════════════════════════════════════════╗
# ║          ★ 每次运行前修改下面的参数 ★              ║
# ╚══════════════════════════════════════════════════════╝

if __name__ == "__main__":
    # ★ 这次要处理的子文件夹（改成你想要处理的路径）
    TARGET_FOLDER = r"音频路径"

    # ★ 最终输出的 JSON（所有子文件夹的结果都会汇集到这里）
    OUTPUT_JSON = r"json路径"

    # 分析采样率
    TARGET_SR = 32000

    # 支持的音频扩展名
    AUDIO_EXTENSIONS = ['.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac', '.wma']

    # ====================================================#
    # 以下为执行代码，一般无需修改                        #
    # ====================================================#

    target_dir = Path(TARGET_FOLDER).resolve()
    if not target_dir.is_dir():
        logger.error(f"目标文件夹不存在: {target_dir}")
        exit(1)

    # 收集本次要处理的音频文件
    audio_files = []
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(target_dir.glob(f"*{ext}"))
        audio_files.extend(target_dir.glob(f"*{ext.upper()}"))
    audio_files = sorted(set(audio_files))

    if not audio_files:
        logger.error(f"在 {target_dir} 中未找到音频文件")
        exit(1)

    logger.info(f"找到 {len(audio_files)} 个音频文件，开始分析...")

    # 读取已有的输出 JSON（如果存在），并建立 file_path -> 条目 的映射
    existing_map = {}
    output_path = Path(OUTPUT_JSON)
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            existing_list = json.load(f)
        for item in existing_list:
            existing_map[item["file_path"]] = item
        logger.info(f"已加载 {len(existing_list)} 条已有记录")
    else:
        logger.info("输出 JSON 不存在，将创建新文件")

    # 分析并合并
    for ap in audio_files:
        logger.info(f"分析 {ap.name} ...")
        result = analyze_file(ap, TARGET_SR)
        # 用 file_path 作为键更新即可（如果已存在则覆盖更新）
        existing_map[result["file_path"]] = result

    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(list(existing_map.values()), f, ensure_ascii=False, indent=2)

    logger.info(f"✅ 本次完成 {len(audio_files)} 个文件，输出 JSON 共 {len(existing_map)} 条记录")