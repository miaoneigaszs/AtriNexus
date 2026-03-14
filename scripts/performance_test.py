"""
性能测试脚本：测量三层记忆串行 vs 并行加载的耗时对比

运行方式：
    cd d:/learnsomething/AtriNexus
    python scripts/performance_test.py

测试内容：
    1. 串行加载：build_full_context()
    2. 并行加载：build_full_context_async()

测试次数：默认 10 次，取平均值
"""

import sys
import os
import time
import asyncio
import random
from datetime import datetime

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.memory_manager import MemoryManager
from src.services.database import Session, MemorySnapshot, init_db
from data.config import config


def prepare_test_data(user_id: str, avatar_name: str, num_memories: int = 20):
    """准备测试数据：短期记忆 + 核心记忆 + 向量记忆"""
    print(f"\n📝 准备测试数据...")
    
    session = Session()
    
    # 1. 准备短期记忆（10轮对话）
    short_memory = []
    for i in range(10):
        short_memory.append({
            "user": f"这是第 {i+1} 条用户消息，关于话题{random.choice(['天气', '工作', '学习', '生活'])}",
            "bot": f"这是第 {i+1} 条AI回复，很高兴和你聊天。",
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    
    # 保存短期记忆
    import json
    snapshot = session.query(MemorySnapshot).filter_by(
        user_id=user_id, avatar_name=avatar_name, memory_type='short'
    ).first()
    if snapshot:
        snapshot.content = json.dumps(short_memory, ensure_ascii=False)
    else:
        snapshot = MemorySnapshot(
            user_id=user_id, avatar_name=avatar_name,
            memory_type='short', content=json.dumps(short_memory, ensure_ascii=False)
        )
        session.add(snapshot)
    
    # 2. 准备核心记忆
    core_memory_content = f"""【稳定信息】
用户是测试用户，喜欢技术讨论，对AI领域感兴趣。

【近期信息】
最近在测试系统性能，关注记忆系统的效率优化。"""
    
    core_data = json.dumps({
        'content': core_memory_content,
        'updated_at': datetime.now().isoformat()
    }, ensure_ascii=False)
    
    core_snapshot = session.query(MemorySnapshot).filter_by(
        user_id=user_id, avatar_name=avatar_name, memory_type='core'
    ).first()
    if core_snapshot:
        core_snapshot.content = core_data
    else:
        core_snapshot = MemorySnapshot(
            user_id=user_id, avatar_name=avatar_name,
            memory_type='core', content=core_data
        )
        session.add(core_snapshot)
    
    session.commit()
    session.close()
    
    print(f"   ✓ 短期记忆: 10 轮对话")
    print(f"   ✓ 核心记忆: 已写入")
    
    # 3. 准备向量记忆（中期记忆）
    memory_manager = MemoryManager()
    
    # 添加一些向量记忆
    summaries = [
        "用户询问了关于 Python 异步编程的问题，讨论了 asyncio 的使用场景。",
        "用户分享了自己学习 RAG 的经历，提到了向量检索的优化思路。",
        "用户讨论了 LLM 应用的成本控制，特别关注 Token 消耗的优化方法。",
        "用户询问了企业微信机器人的开发流程，包括消息回调的处理方式。",
        "用户分享了项目部署的经验，讨论了 Docker 容器化的最佳实践。",
    ]
    
    for summary in summaries[:num_memories]:
        memory_manager.add_to_vector_memory(user_id, avatar_name, summary)
    
    print(f"   ✓ 向量记忆: {num_memories} 条摘要")
    print("✅ 测试数据准备完成\n")
    
    return memory_manager


def test_serial_loading(memory_manager, user_id: str, avatar_name: str, query: str):
    """测试串行加载"""
    start = time.perf_counter()
    result = memory_manager.build_full_context(user_id, avatar_name, query)
    elapsed = (time.perf_counter() - start) * 1000  # 转换为毫秒
    return elapsed, result


async def test_parallel_loading(memory_manager, user_id: str, avatar_name: str, query: str):
    """测试并行加载"""
    start = time.perf_counter()
    result = await memory_manager.build_full_context_async(user_id, avatar_name, query)
    elapsed = (time.perf_counter() - start) * 1000  # 转换为毫秒
    return elapsed, result


def run_performance_test(runs: int = 10):
    """运行性能测试"""
    print("=" * 60)
    print("🚀 AtriNexus 三层记忆性能测试")
    print("=" * 60)
    
    user_id = "perf_test_user"
    avatar_name = "test_avatar"
    query = "我想了解一下项目的性能优化方法"
    
    # 准备测试数据
    memory_manager = prepare_test_data(user_id, avatar_name)
    
    print("📊 开始性能测试...")
    print(f"   测试次数: {runs} 次")
    print("-" * 60)
    
    # 测试串行加载
    serial_times = []
    print("\n🔄 测试串行加载 (build_full_context)...")
    for i in range(runs):
        elapsed, _ = test_serial_loading(memory_manager, user_id, avatar_name, query)
        serial_times.append(elapsed)
        print(f"   第 {i+1:2d} 次: {elapsed:7.2f} ms")
    
    # 测试并行加载
    parallel_times = []
    print("\n⚡ 测试并行加载 (build_full_context_async)...")
    for i in range(runs):
        elapsed, _ = asyncio.run(test_parallel_loading(memory_manager, user_id, avatar_name, query))
        parallel_times.append(elapsed)
        print(f"   第 {i+1:2d} 次: {elapsed:7.2f} ms")
    
    # 计算统计数据
    serial_avg = sum(serial_times) / len(serial_times)
    parallel_avg = sum(parallel_times) / len(parallel_times)
    serial_min = min(serial_times)
    parallel_min = min(parallel_times)
    serial_max = max(serial_times)
    parallel_max = max(parallel_times)
    
    # 计算提升百分比
    improvement = ((serial_avg - parallel_avg) / serial_avg) * 100
    
    # 输出结果
    print("\n" + "=" * 60)
    print("📈 测试结果汇总")
    print("=" * 60)
    
    print(f"\n📊 串行加载 (build_full_context):")
    print(f"   平均耗时: {serial_avg:7.2f} ms")
    print(f"   最小耗时: {serial_min:7.2f} ms")
    print(f"   最大耗时: {serial_max:7.2f} ms")
    
    print(f"\n⚡ 并行加载 (build_full_context_async):")
    print(f"   平均耗时: {parallel_avg:7.2f} ms")
    print(f"   最小耗时: {parallel_min:7.2f} ms")
    print(f"   最大耗时: {parallel_max:7.2f} ms")
    
    print(f"\n🚀 性能提升:")
    print(f"   平均提升: {improvement:.1f}%")
    print(f"   耗时减少: {serial_avg - parallel_avg:.2f} ms")
    
    print("\n" + "=" * 60)
    print("💡 简历建议数据:")
    print("=" * 60)
    print(f'   "通过 asyncio.gather 并行加载，耗时降低约 {improvement:.0f}%"')
    print(f'   "串行 ~{serial_avg:.0f}ms → 并行 ~{parallel_avg:.0f}ms"')
    
    # 清理测试数据
    print("\n🧹 清理测试数据...")
    session = Session()
    session.query(MemorySnapshot).filter_by(user_id=user_id).delete()
    session.commit()
    session.close()
    
    # 删除向量记忆
    try:
        memory_manager.delete_vector_memory(user_id, avatar_name)
        print("   ✓ 测试数据已清理")
    except:
        print("   ⚠ 向量数据需手动清理")
    
    return {
        "serial_avg": serial_avg,
        "parallel_avg": parallel_avg,
        "improvement": improvement
    }


if __name__ == "__main__":
    # 运行测试
    result = run_performance_test(runs=10)
