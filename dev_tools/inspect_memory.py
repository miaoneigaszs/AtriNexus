import os
import sys
import json
import sqlite3
from typing import Optional

# 确保能找到 src 模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

try:
    import chromadb
except ImportError:
    chromadb = None


def inspect_sqlite_memory(user_id: Optional[str] = None):
    """检查 chat_history.db 中的短期和核心记忆"""
    db_path = os.path.join(project_root, 'data', 'database', 'chat_history.db')
    if not os.path.exists(db_path):
        print(f"❌ 找不到数据库文件: {db_path}")
        return

    print("\n" + "="*50)
    print("🧠 SQLite 短期/核心记忆检查")
    print("="*50)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    query = "SELECT user_id, avatar_name, memory_type, content, updated_at FROM memory_snapshots"
    params = []
    if user_id:
        query += " WHERE user_id = ?"
        params.append(user_id)
        
    cursor.execute(query, params)
    rows = cursor.fetchall()
    
    if not rows:
        print("暂无由于对话产生的记忆快照。")
        return
        
    for row in rows:
        uid, avatar, mem_type, content, updated_at = row
        print(f"[{updated_at}] User: {uid} | Avatar: {avatar} | Type: {mem_type}")
        try:
            parsed = json.loads(content)
            print(json.dumps(parsed, indent=2, ensure_ascii=False))
        except:
            print(content)
        print("-" * 30)
        
    conn.close()


def inspect_chromadb_memory(user_id: Optional[str] = None):
    """检查 ChromaDB 中的向量检索中期记忆"""
    print("\n" + "="*50)
    print("🌌 ChromaDB 向量中期记忆检查")
    print("="*50)
    
    if not chromadb:
        print("❌ 未安装 chromadb 库，无法查看。")
        return
        
    db_path = os.path.join(project_root, 'data', 'vectordb')
    if not os.path.exists(db_path):
        print(f"❌ 找不到 ChromaDB 目录: {db_path}")
        return
        
    client = chromadb.PersistentClient(path=db_path)
    collections = client.list_collections()
    
    if not collections:
        print("暂无向量集合。")
        return
        
    for collection_meta in collections:
        col = client.get_collection(collection_meta.name)
        count = col.count()
        print(f"\n📂 集合名称: {collection_meta.name} (包含 {count} 条记忆)")
        
        if count > 0:
            results = col.get()
            docs = results['documents']
            metas = results['metadatas']
            
            for i in range(count):
                meta = metas[i] if metas else {}
                # 条件过滤
                if user_id and meta.get("user_id") != user_id:
                    continue
                    
                doc = docs[i] if docs else ""
                print(f"  [时间: {meta.get('timestamp', 'N/A')}] User: {meta.get('user_id')} | Avatar: {meta.get('avatar_name')}")
                print(f"  摘要: {doc}")
                print("  " + "-"*40)


if __name__ == "__main__":
    print("AtriNexus 记忆检查工具")
    
    # 你可以传入你在调试用的 UserID，比如 "WangZhi"
    target_user = input("输入要查看的企微 UserID (直接回车查看所有): ").strip()
    target_user = target_user if target_user else None
    
    inspect_sqlite_memory(target_user)
    inspect_chromadb_memory(target_user)
