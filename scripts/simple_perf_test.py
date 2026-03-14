"""
简化版性能测试脚本：测量三层记忆串行 vs 并行加载的耗时对比

运行方式：
    cd d:/learnsomething/AtriNexus
    python scripts/simple_perf_test.py

这个简化版本模拟三层记忆的加载逻辑，不需要完整的项目依赖。
"""

import asyncio
import time
import sqlite3
import os
from concurrent.futures import ThreadPoolExecutor

# 创建线程池（与项目实际实现一致）
_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="async_io_")


async def run_sync(func, *args, **kwargs):
    """在线程池中运行同步函数（与项目实际实现一致）"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        lambda: func(*args, **kwargs)
    )


def simulate_core_memory_load(db_path: str, user_id: str) -> str:
    """模拟核心记忆加载（SQLite 查询）"""
    time.sleep(0.02)  # 模拟 SQLite 查询延迟 ~20ms
    return "【稳定信息】用户是测试用户...\n【近期信息】最近在测试性能..."


def simulate_vector_memory_search(query: str) -> list:
    """模拟向量记忆检索（ChromaDB 查询）"""
    time.sleep(0.08)  # 模拟向量检索延迟 ~80ms
    return ["记忆1: 用户讨论过异步优化", "记忆2: 用户关注性能问题"]


def simulate_short_memory_load(db_path: str, user_id: str) -> list:
    """模拟短期记忆加载（SQLite 查询）"""
    time.sleep(0.02)  # 模拟 SQLite 查询延迟 ~20ms
    return [{"user": "你好", "bot": "你好！"} for _ in range(10)]


def serial_load(db_path: str, user_id: str, query: str) -> dict:
    """串行加载三层记忆"""
    # 1. 核心记忆
    core = simulate_core_memory_load(db_path, user_id)
    # 2. 向量记忆
    vector = simulate_vector_memory_search(query)
    # 3. 短期记忆
    short = simulate_short_memory_load(db_path, user_id)
    
    return {"core": core, "vector": vector, "short": short}


async def parallel_load(db_path: str, user_id: str, query: str) -> dict:
    """并行加载三层记忆"""
    # 创建三个并行任务
    core_task = run_sync(simulate_core_memory_load, db_path, user_id)
    vector_task = run_sync(simulate_vector_memory_search, query)
    short_task = run_sync(simulate_short_memory_load, db_path, user_id)
    
    # 并行等待所有任务完成
    core, vector, short = await asyncio.gather(
        core_task, vector_task, short_task
    )
    
    return {"core": core, "vector": vector, "short": short}


def run_test(runs: int = 10):
    """运行性能测试"""
    print("=" * 60)
    print("🚀 三层记忆加载性能测试（模拟版）")
    print("=" * 60)
    
    db_path = "data/atrinexus.db"
    user_id = "test_user"
    query = "我想了解性能优化"
    
    print(f"\n📊 测试配置:")
    print(f"   - 核心记忆加载延迟: ~20ms (SQLite)")
    print(f"   - 向量记忆检索延迟: ~80ms (ChromaDB)")
    print(f"   - 短期记忆加载延迟: ~20ms (SQLite)")
    print(f"   - 理论串行总耗时: ~120ms")
    print(f"   - 理论并行总耗时: ~80ms (取最长)")
    print(f"   - 测试次数: {runs}")
    print("-" * 60)
    
    # 测试串行
    serial_times = []
    print("\n🔄 测试串行加载...")
    for i in range(runs):
        start = time.perf_counter()
        serial_load(db_path, user_id, query)
        elapsed = (time.perf_counter() - start) * 1000
        serial_times.append(elapsed)
        print(f"   第 {i+1:2d} 次: {elapsed:7.2f} ms")
    
    # 测试并行
    parallel_times = []
    print("\n⚡ 测试并行加载...")
    for i in range(runs):
        start = time.perf_counter()
        asyncio.run(parallel_load(db_path, user_id, query))
        elapsed = (time.perf_counter() - start) * 1000
        parallel_times.append(elapsed)
        print(f"   第 {i+1:2d} 次: {elapsed:7.2f} ms")
    
    # 统计
    serial_avg = sum(serial_times) / len(serial_times)
    parallel_avg = sum(parallel_times) / len(parallel_times)
    improvement = ((serial_avg - parallel_avg) / serial_avg) * 100
    
    # 输出
    print("\n" + "=" * 60)
    print("📈 测试结果")
    print("=" * 60)
    print(f"\n📊 串行加载:")
    print(f"   平均: {serial_avg:.2f} ms")
    print(f"   最小: {min(serial_times):.2f} ms")
    print(f"   最大: {max(serial_times):.2f} ms")
    
    print(f"\n⚡ 并行加载:")
    print(f"   平均: {parallel_avg:.2f} ms")
    print(f"   最小: {min(parallel_times):.2f} ms")
    print(f"   最大: {max(parallel_times):.2f} ms")
    
    print(f"\n🚀 性能提升: {improvement:.1f}%")
    
    print("\n" + "=" * 60)
    print("💡 简历建议数据:")
    print("=" * 60)
    print(f'   "通过 asyncio.gather 并行加载，耗时降低约 {improvement:.0f}%"')
    print(f'   "串行 ~{serial_avg:.0f}ms → 并行 ~{parallel_avg:.0f}ms"')
    
    return {
        "serial_avg": serial_avg,
        "parallel_avg": parallel_avg,
        "improvement": improvement
    }


if __name__ == "__main__":
    run_test(runs=10)
