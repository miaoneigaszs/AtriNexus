"""
表情包多模态描述生成脚本
使用阿里云 DashScope (Qwen VL) 视觉模型自动为表情包图片生成描述和标签。
生成结果存储在 data/mode/emoji_descriptions.json 中。

使用方法:
  1. 在下方 API_KEY 处填入你的阿里云百炼 API Key
  2. 运行: python scripts/describe_emojis.py
"""

import os
import sys
import json
import base64
import time
import logging
from pathlib import Path
from openai import OpenAI

# ======================== 配置区 ========================
API_KEY = "sk-17114f341b004379950e49ee99145de7"  # 在这里填入你的阿里云百炼 API Key，例如 "sk-xxx"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen3-vl-flash"  # 阿里云免费视觉模型
# ========================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 项目根目录
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_MODE_DIR = os.path.join(ROOT_DIR, "data", "mode")
OUTPUT_FILE = os.path.join(DATA_MODE_DIR, "emoji_descriptions.json")

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

PROMPT = """请用简短的中文描述这张表情包图片的内容和情感。
要求：
1. 一句话描述画面内容（不超过30字）
2. 给出1-3个情感标签，用逗号分隔（如：开心,搞笑,可爱）
格式：
描述：<一句话描述>
标签：<标签1>,<标签2>"""


def encode_image_to_base64(image_path: str) -> str:
    """将本地图片编码为 base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_mime(filepath: str) -> str:
    """根据扩展名返回 MIME 类型"""
    ext = os.path.splitext(filepath)[1].lower()
    mime_map = {
        '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
        '.png': 'image/png', '.gif': 'image/gif',
        '.bmp': 'image/bmp', '.webp': 'image/webp',
    }
    return mime_map.get(ext, 'image/jpeg')


def describe_image(client: OpenAI, image_path: str, max_retries=3) -> dict:
    """调用 Qwen VL 模型描述一张图片，带重试"""
    b64 = encode_image_to_base64(image_path)
    mime = get_image_mime(image_path)
    data_url = f"data:{mime};base64,{b64}"

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": PROMPT},
                        ],
                    }
                ],
            )
            raw = completion.choices[0].message.content.strip()

            # 解析结果
            description = raw
            tags = []
            for line in raw.split('\n'):
                line = line.strip()
                if line.startswith('描述') and '：' in line:
                    description = line.split('：', 1)[1].strip()
                elif line.startswith('标签') and '：' in line:
                    tags = [t.strip() for t in line.split('：', 1)[1].split(',')]

            return {"description": description, "tags": tags, "raw": raw}

        except Exception as e:
            wait = 2 ** (attempt + 1)
            logger.warning(f"第 {attempt+1} 次请求失败: {e}，{wait}s 后重试...")
            time.sleep(wait)

    logger.error(f"多次重试后仍失败: {image_path}")
    return None


def main():
    if not API_KEY:
        logger.error("请先在脚本顶部 API_KEY 处填入你的阿里云百炼 API Key！")
        sys.exit(1)

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # 加载已有结果（支持断点续传）
    existing = {}
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            logger.info(f"已加载 {len(existing)} 条已有描述，将跳过这些文件")
        except Exception:
            pass

    # 扫描图片
    images = []
    for root, dirs, files in os.walk(DATA_MODE_DIR):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                images.append(os.path.join(root, f))

    logger.info(f"共扫描到 {len(images)} 张图片")

    processed = 0
    skipped = 0
    for i, img_path in enumerate(images):
        filename = os.path.basename(img_path)
        rel_path = os.path.relpath(img_path, DATA_MODE_DIR)

        if filename in existing:
            skipped += 1
            continue

        logger.info(f"[{i+1}/{len(images)}] 处理: {filename}")

        result = describe_image(client, img_path)
        if result:
            existing[filename] = {
                "path": rel_path,
                "description": result["description"],
                "tags": result.get("tags", []),
                "generated_at": time.time(),
            }
            processed += 1
            logger.info(f"  ✅ {result['description']}")

            # 每处理 5 张保存一次（防丢失）
            if processed % 5 == 0:
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(existing, f, ensure_ascii=False, indent=2)

            # 请求间隔，避免限流
            time.sleep(0.5)
        else:
            logger.error(f"  ❌ 跳过: {filename}")

    # 最终保存
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    logger.info(f"\n完成！处理 {processed} 张，跳过 {skipped} 张。结果保存至 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
