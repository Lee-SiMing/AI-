import os
import json
import torch
import librosa
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from panns_inference import AudioTagging

# ==========================================
# CONFIG
# ==========================================

ANCHOR_JSON = "参考json路径"   # 参考锚点库
TARGET_JSON = "中国调式转换json路径"      # 之前生成的 JSON（待增强）
NEW_AUDIO_DIR = "音频路径"                    # 新音频文件夹
OUTPUT_JSON = "输出路径"         # 输出（建议先不要覆盖原文件）

CHECKPOINT_PATH = r"C:\Users\Administrator\panns_data\Cnn14_mAP=0.431.pth"
TOP_K = 5

# ==========================================
# LOAD ANCHORS
# ==========================================

with open(ANCHOR_JSON, "r", encoding="utf-8") as f:
    anchors = json.load(f)

print(f"✅ Loaded {len(anchors)} anchors")

# ==========================================
# LOAD TARGET JSON
# ==========================================

with open(TARGET_JSON, "r", encoding="utf-8") as f:
    target_data = json.load(f)

print(f"✅ Loaded {len(target_data)} target entries")

# 建立双重映射：绝对路径 -> 条目，文件名 -> 条目（用于路径匹配失败时的回退）
target_map_abs = {}
target_map_by_filename = {}

for item in target_data:
    fp = item.get("file_path", "")
    if not fp:
        continue

    # 尝试转为绝对路径
    abs_fp = os.path.abspath(fp)
    target_map_abs[abs_fp] = item

    # 同时也保存文件名（用于后备匹配）
    filename = os.path.basename(fp)
    target_map_by_filename[filename] = item

# ==========================================
# LOAD PANNs
# ==========================================

device = "cuda" if torch.cuda.is_available() else "cpu"
at = AudioTagging(checkpoint_path=CHECKPOINT_PATH, device=device)

# ==========================================
# EMBEDDING FUNCTION
# ==========================================

def extract_embedding(audio_path):
    y, _ = librosa.load(audio_path, sr=32000, mono=True)
    y = y[None, :]                 # batch 维度
    _, embedding = at.inference(y)
    return embedding.flatten()     # (2048,)

# ==========================================
# COLLECT AUDIO FILES
# ==========================================

audio_files = []
for root, _, files in os.walk(NEW_AUDIO_DIR):
    for file in files:
        if file.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
            audio_files.append(os.path.abspath(os.path.join(root, file)))

print(f"✅ Found {len(audio_files)} audio files to process\n")

# ==========================================
# PROCESS
# ==========================================

for audio_path in tqdm(audio_files, desc="Tagging"):
    # ---------- 查找目标条目 ----------
    target_item = target_map_abs.get(audio_path)               # 优先绝对路径匹配
    if target_item is None:
        filename = os.path.basename(audio_path)
        target_item = target_map_by_filename.get(filename)     # 后备：仅文件名匹配

    if target_item is None:
        tqdm.write(f"⚠️ SKIP (not in target JSON): {audio_path}")
        continue

    # ---------- 提取 embedding ----------
    try:
        new_emb = extract_embedding(audio_path)
    except Exception as e:
        tqdm.write(f"❌ EMBEDDING ERROR: {audio_path} - {e}")
        continue

    # ---------- 与锚点库比较 ----------
    similarities = []
    for anchor in anchors:
        try:
            anchor_emb = np.array(anchor["features"]["embedding"])
            sim = cosine_similarity([new_emb], [anchor_emb])[0][0]
            similarities.append({
                "song_id": anchor["song_id"],
                "file_path": anchor["file_path"],
                "similarity": float(sim),
                "instrument": anchor["features"].get("instrument", []),
                "genre": anchor["features"].get("genre", []),
                "mood": anchor["features"].get("mood", []),
                "vibe": anchor["features"].get("vibe", [])
            })
        except Exception as e:
            tqdm.write(f"❌ COMPARE ERROR: {e}")

    # 排序 & top-k
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    top_matches = similarities[:TOP_K]

    # ---------- 合并标签 ----------
    inst_tags = []
    genre_tags = []
    mood_tags = []
    vibe_tags = []
    for m in top_matches:
        inst_tags.extend(m["instrument"])
        genre_tags.extend(m["genre"])
        mood_tags.extend(m["mood"])
        vibe_tags.extend(m["vibe"])

    inst_tags = list(set(inst_tags))
    genre_tags = list(set(genre_tags))
    mood_tags = list(set(mood_tags))
    vibe_tags = list(set(vibe_tags))

    # ---------- 写入目标条目 ----------
    if "features" not in target_item:
        target_item["features"] = {}

    features = target_item["features"]

    # 合并原有标签（如果有）与预测标签
    features["instrument"] = list(set(features.get("instrument", []) + inst_tags))
    features["genre"]      = list(set(features.get("genre", []) + genre_tags))
    features["mood"]       = list(set(features.get("mood", []) + mood_tags))
    features["vibe"]       = list(set(features.get("vibe", []) + vibe_tags))

    # 同时保存 top_matches 供查验
    features["top_matches"] = top_matches

    tqdm.write(f"✔ Updated: {audio_path}")

# ==========================================
# SAVE
# ==========================================

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(target_data, f, ensure_ascii=False, indent=2)

print(f"\n🎉 DONE! Output saved to: {OUTPUT_JSON}")