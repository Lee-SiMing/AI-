import os
import json
import librosa
from tqdm import tqdm
from panns_inference import AudioTagging

# ==============================
# ★ 每次运行前修改以下配置 ★
# ==============================
INPUT_JSON = "中国调式json路径"   # 包含所有文件 bpm/key 的 JSON
OUTPUT_JSON = "输出路径"   # 最终累积输出文件
CHECKPOINT_PATH = r"C:/Users/Administrator/panns_data/Cnn14_mAP=0.431.pth"

TARGET_FOLDER = r"音频路径"       # 本次要处理的子文件夹

# 这个文件夹的统一标签
INSTRUMENT_TAGS = ["xiao"]    # 示例，根据实际修改
GENRE_TAGS      = ["Chinese ethnical"]
MOOD_TAGS       = ["Relaxed / Chill"]
VIBE_TAGS       = ["Laid-back"]

# ==============================
# 辅助函数：路径归一化
# ==============================
def normalize_path(p):
    """将路径转为统一的绝对路径，消除正反斜杠差异"""
    return os.path.normcase(os.path.abspath(p))

# ==============================
# 1. 加载 INPUT_JSON（基础特征库）
# ==============================
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    base_list = json.load(f)

base_map = {}
for item in base_list:
    fp = item.get("file_path")
    if fp:
        base_map[normalize_path(fp)] = item

print(f"✓ 从 INPUT_JSON 加载了 {len(base_map)} 条基础记录")

# ==============================
# 2. 加载模型（只做一次）
# ==============================
device = "cuda" if __import__('torch').cuda.is_available() else "cpu"
at = AudioTagging(checkpoint_path=CHECKPOINT_PATH, device=device)

def extract_embedding(audio_path):
    audio, _ = librosa.load(audio_path, sr=32000, mono=True)
    audio = audio[None, :]
    _, embedding = at.inference(audio)
    return embedding.flatten().tolist()

# ==============================
# 3. 读取已有的输出 JSON（防止丢失历史数据）
# ==============================
if os.path.exists(OUTPUT_JSON):
    with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
        existing_items = json.load(f)
    existing_map = {normalize_path(it["file_path"]): it for it in existing_items}
    print(f"✓ 已加载 {len(existing_items)} 条已有输出记录")
else:
    existing_map = {}
    print("○ 输出 JSON 不存在，将创建新文件")

# ==============================
# 4. 收集本次要处理的音频文件（只来自 TARGET_FOLDER）
# ==============================
audio_files = []
for root, _, files in os.walk(TARGET_FOLDER):
    for file in files:
        if file.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
            audio_files.append(os.path.abspath(os.path.join(root, file)))

print(f"→ 本次处理 {len(audio_files)} 个文件，来自：{TARGET_FOLDER}")

# ==============================
# 5. 逐个处理并合并
# ==============================
for abs_path in tqdm(audio_files, desc="Processing"):
    norm_path = normalize_path(abs_path)

    # 从基础库中获取条目（深拷贝，避免修改原数据）
    base_item = base_map.get(norm_path)
    if base_item:
        item = json.loads(json.dumps(base_item))
    else:
        # 如果基础库里没有这个文件，创建一个新条目（保留路径，特征留空）
        item = {
            "song_id": os.path.splitext(os.path.basename(abs_path))[0],
            "file_path": abs_path,
            "title": os.path.splitext(os.path.basename(abs_path))[0],
            "features": {}
        }
        tqdm.write(f"⚠ 注意：{abs_path} 未在 INPUT_JSON 中找到，已创建新条目（缺少 bpm/key）")

    # 确保 features 字段存在
    if "features" not in item:
        item["features"] = {}

    # 提取 embedding
    try:
        embedding = extract_embedding(abs_path)
    except Exception as e:
        tqdm.write(f"✗ EMBEDDING 错误 ({abs_path}): {e}")
        continue

    # 添加手工标签
    item["features"]["instrument"] = INSTRUMENT_TAGS
    item["features"]["genre"]      = GENRE_TAGS
    item["features"]["mood"]       = MOOD_TAGS
    item["features"]["vibe"]       = VIBE_TAGS
    item["features"]["embedding"]  = embedding

    # 更新到合并字典（以归一化路径为键）
    existing_map[norm_path] = item

# ==============================
# 6. 保存最终的 JSON（包含所有历史数据）
# ==============================
final_list = list(existing_map.values())
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(final_list, f, ensure_ascii=False, indent=2)

print(f"\n✅ 本次处理 {len(audio_files)} 个文件")
print(f"✅ 输出文件 '{OUTPUT_JSON}' 现共有 {len(final_list)} 条记录")