#!/usr/bin/env python3
"""
音乐提示词泛化脚本（调式和速度独立展示）
输入：包含音乐片段元数据的 JSON 文件（支持嵌套 features 结构）
输出：包含 sound_id, file_path, generalized_prompt 的 JSON 文件
"""

import os
import json
import time
from typing import List, Dict, Any, Optional

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("请先安装 openai 库: pip install openai")


def load_json(file_path: str) -> Any:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def resolve_absolute_path(relative_path: str, base_dir: Optional[str] = None) -> str:
    """将路径转换为绝对路径，若已是绝对路径则直接返回"""
    if os.path.isabs(relative_path):
        return relative_path
    if base_dir:
        return os.path.normpath(os.path.join(base_dir, relative_path))
    else:
        return os.path.abspath(relative_path)


def extract_metadata(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    从输入对象中提取所需元数据。
    支持两种结构：
      1. 扁平结构（旧版）：直接包含 instrument, genre, mood, vibe 等字段
      2. 嵌套结构（新版）：包含 features 子对象，其中含有上述字段
    """
    if "features" in item and isinstance(item["features"], dict):
        features = item["features"]
        instrument = features.get("instrument", [])
        genre = features.get("genre", [])
        mood = features.get("mood", [])
        vibe = features.get("vibe", [])
        bpm = features.get("bpm")
        key = features.get("key")
    else:
        instrument = item.get("instrument", [])
        genre = item.get("genre", [])
        mood = item.get("mood", [])
        vibe = item.get("vibe", [])
        bpm = item.get("bpm")
        key = item.get("key")

    similarity = item.get("similarity")
    if similarity is None and "features" in item:
        similarity = item["features"].get("similarity")

    return {
        "instrument": instrument,
        "genre": genre,
        "mood": mood,
        "vibe": vibe,
        "bpm": bpm,
        "key": key,
        "similarity": similarity,
    }


def build_prompt(metadata: Dict[str, Any]) -> str:
    """根据元数据构建 DeepSeek 提示词（调式、速度独立展示）"""
    instruments = ", ".join(metadata["instrument"]) if metadata["instrument"] else "unknown instruments"
    genres = ", ".join(metadata["genre"]) if metadata["genre"] else "unknown genre"
    moods = ", ".join(metadata["mood"]) if metadata["mood"] else "neutral"
    vibes = ", ".join(metadata["vibe"]) if metadata["vibe"] else "neutral"

    # 显式处理调式和速度
    bpm_str = metadata.get("bpm", "unknown")
    key_str = metadata.get("key", "unknown")

    # 其他附加信息（如相似度）
    extra = []
    if metadata.get("similarity") is not None:
        extra.append(f"similarity score: {metadata['similarity']:.4f}")
    extra_text = f" ({'; '.join(extra)})" if extra else ""

    prompt = f"""You are a professional music prompt engineer. Given the following metadata of a music clip:
Instruments: {instruments}
Genre: {genres}
Mood: {moods}
Vibe: {vibes}
Tempo: {bpm_str}
Key/Mode: {key_str}{extra_text}

Generate a short but expressive generalized music prompt in English (about 30-80 words) to guide AI to generate new music of a similar style. The prompt should combine the above features and provide a vivid, imagery-driven or emotion-guided description. Do not mention specific song names or file names. Output only the prompt itself, without any extra explanation."""
    return prompt


def call_deepseek(client: OpenAI, model: str, prompt: str, max_retries: int, retry_delay: int) -> str:
    """调用 DeepSeek API 生成提示词，带重试逻辑"""
    messages = [
        {"role": "system", "content": "You are an assistant that transforms music tags into vivid English prompts."},
        {"role": "user", "content": prompt}
    ]
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise RuntimeError(f"经过 {max_retries} 次重试仍无法获取结果: {e}")
    return ""


def main():
    # ==================== 配置参数（请按需修改） ====================
    INPUT_JSON_PATH = "E:输入路径"  # 输入 JSON 文件路径
    OUTPUT_JSON_PATH = "E:输出路径"  # 输出 JSON 文件路径
    DEEPSEEK_API_KEY = "你的APIKEY"  # DeepSeek API Key
    MODEL_NAME = "你的模型"  # 模型名称，例如 deepseek-chat
    BASE_URL = "APIURL"  # API Base URL
    MAX_RETRIES = 3  # 请求失败最大重试次数
    RETRY_DELAY = 2  # 重试间隔（秒）
    BASE_DIR = None  # 相对路径基准目录（None 表示使用输入文件所在目录）
    # ================================================================

    # 设置 API Key 环境变量或直接使用（脚本中直接使用变量）
    api_key = DEEPSEEK_API_KEY
    if not api_key or api_key == "your-api-key-here":
        raise ValueError("请在脚本末尾配置 DEEPSEEK_API_KEY 为有效的 API Key")

    # 初始化客户端
    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    # 确定相对路径基准目录
    base_dir = BASE_DIR
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(INPUT_JSON_PATH))

    # 加载输入 JSON
    input_data = load_json(INPUT_JSON_PATH)

    # 兼容单对象或数组
    if isinstance(input_data, dict):
        items = [input_data]
    elif isinstance(input_data, list):
        items = input_data
    else:
        raise TypeError("输入 JSON 必须是对象或对象数组")

    output_items = []
    total = len(items)
    for idx, item in enumerate(items, start=1):
        song_id = item.get("song_id")
        raw_path = item.get("file_path")
        if not song_id or not raw_path:
            print(f"第 {idx} 条：缺少 song_id 或 file_path，跳过")
            continue

        print(f"处理第 {idx}/{total} 条: {song_id}")
        try:
            metadata = extract_metadata(item)
            user_prompt = build_prompt(metadata)
            generalized = call_deepseek(client, MODEL_NAME, user_prompt, MAX_RETRIES, RETRY_DELAY)
            abs_path = resolve_absolute_path(raw_path, base_dir)

            output_items.append({
                "sound_id": song_id,
                "file_path": abs_path,
                "generalized_prompt": generalized
            })
            print(f"  -> 提示词: {generalized[:60]}...")
        except Exception as e:
            print(f"  处理失败: {e}")
            abs_path = resolve_absolute_path(raw_path, base_dir)
            output_items.append({
                "sound_id": song_id,
                "file_path": abs_path,
                "generalized_prompt": f"ERROR: {str(e)}"
            })

    save_json(output_items, OUTPUT_JSON_PATH)
    print(f"处理完成，共 {len(output_items)} 条记录，已保存到: {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()