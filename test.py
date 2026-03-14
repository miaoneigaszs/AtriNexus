import requests
import json
from pprint import pprint  # 导入格式化打印工具

base_url = "https://api.siliconflow.cn/v1/embeddings"
api_key = "sk-iajerqwxwqfrsfoflscyyplastwdgzvufabikztazgmlxwnt"
model = "BAAI/bge-m3"

doc = [
    "这是一个测试文档",
    "这是第二行测试文档",
    "这是第三行测试文档"
]

payload = {
    "model": model,
    "input": doc,
    "encoding_format": "float"  # 修正拼写错误
}

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

response = requests.post(base_url, headers=headers, json=payload)
response.raise_for_status()
result = response.json()


data = result.get("data", [])
print("=== Data 部分整体信息 ===")
print(f"✅ data 列表长度（元素个数）：{len(data)}")
print(f"✅ data 类型：{type(data)}")

if data:  # 如果data非空
    print("\n=== 第一条数据的完整结构（向量只显示长度）===")
    first_item = data[0].copy()  # 复制第一条数据，避免修改原数据
    # 把长向量替换成「类型+长度」，避免截断
    if "embedding" in first_item:
        emb = first_item["embedding"]
        first_item["embedding"] = f"list (长度: {len(emb)}) → 示例值: {emb[:5]}..."  # 只显示前5个值
    
    pprint(first_item, indent=2)  # 格式化打印结构
    
    # 可选：打印所有数据的基础信息（不打印向量）
    print("\n=== 所有数据的基础信息（仅index和object）===")
    for idx, item in enumerate(data):
        print(f"第{idx}条：index={item.get('index')}, object={item.get('object')}")