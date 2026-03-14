import asyncio
import time
import statistics
import argparse
import sys
import os

# 将项目根目录加入 sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 模拟的 WeComClient，防止真发消息给企微导致轰炸
class MockWeComClient:
    def send_text(self, user_id, content):
        pass

async def init_services():
    from src.wecom.handlers.message_handler import MessageHandler
    
    mock_client = MockWeComClient()
    handler = MessageHandler(wecom_client=mock_client)
    return handler

async def simulate_user_request(handler, user_id, query, msg_id):
    """直接调用核心处理逻辑"""
    start_time = time.time()
    try:
        # 直接调用底层的 _execute_kb_search 来模拟完整的大模型询问+知识库检索+记忆处理
        # 抛弃上游的鉴权或企微队列
        await handler._execute_kb_search(user_id=user_id, content=query, msg_id=msg_id)
        latency = time.time() - start_time
        return {"status": "success", "latency": latency}
    except Exception as e:
        latency = time.time() - start_time
        return {"status": "error", "error": str(e), "latency": latency}

async def run_load_test(concurrency: int, total_requests: int):
    print("⏳ 正在初始化服务和模型客户端，请稍候...")
    handler = await init_services()
    print("✅ 服务初始化完成！")
    
    print(f"\n🚀 开始压测 -> 并发数: {concurrency}, 总请求数: {total_requests}")
    
    test_queries = [
        "你好，我该如何称呼你？",
        "查询一下最近的项目进展情况。",
        "帮我搜索一下今天北京天气怎么样？",
        "公司请病假是怎么规定的？",
        "你还记得我上个问题问的什么吗？"
    ]
    
    start_time = time.time()
    tasks = []
    
    for i in range(total_requests):
        query = test_queries[i % len(test_queries)]
        user_id = f"benchmark_user_{i}"  # 每个请求独立用户，避免用户级锁
        msg_id = f"mock_msg_{int(time.time()*1000)}_{i}"
        
        tasks.append(simulate_user_request(handler, user_id, query, msg_id))

    results = []
    
    # 控制并发批次
    for i in range(0, len(tasks), concurrency):
        batch = tasks[i:i+concurrency]
        batch_results = await asyncio.gather(*batch)
        results.extend(batch_results)
        print(f"📊 已完成 {min(i+concurrency, total_requests)} / {total_requests} 个请求...")

    total_time = time.time() - start_time
    
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "error"]
    latencies = [r["latency"] for r in successful]

    print("\n" + "="*45)
    print("🎯 AtriNexus - 内部核心处理吞吐压测分析")
    print("="*45)
    print(f"总耗时:      {total_time:.2f} 秒")
    print(f"总请求数:    {total_requests}")
    print(f"成功数:      {len(successful)}")
    print(f"失败数:      {len(failed)}")
    
    if successful:
        print(f"\n[性能指标]")
        print(f"吞吐量 (QPS): {len(successful) / total_time:.2f} req/s")
        print(f"平均响应时间: {statistics.mean(latencies):.2f} 秒")
        print(f"最小响应时间: {min(latencies):.2f} 秒")
        print(f"最大响应时间: {max(latencies):.2f} 秒")
        
        if len(latencies) >= 2:
            sorted_latencies = sorted(latencies)
            # 使用 numpy 风格的百分位计算，避免外推超出范围
            def percentile(data, p):
                """计算第 p 百分位数，使用线性插值但不会超出数据范围"""
                k = (len(data) - 1) * p / 100
                f = int(k)
                c = f + 1 if f + 1 < len(data) else f
                return data[f] + (data[c] - data[f]) * (k - f)
            
            print(f"P50 响应时间: {percentile(sorted_latencies, 50):.2f} 秒")
            print(f"P90 响应时间: {percentile(sorted_latencies, 90):.2f} 秒")
            print(f"P95 响应时间: {percentile(sorted_latencies, 95):.2f} 秒")
    
    if failed:
        print("\n❌ 错误原因 (前5条):")
        for f in failed[:5]:
            print(f"  - {f['error']}")
            
    print("="*45)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AtriNexus Internal Benchmark")
    parser.add_argument("-c", "--concurrency", type=int, default=5, help="并发数量")
    parser.add_argument("-n", "--requests", type=int, default=10, help="总请求数量")
    
    args = parser.parse_args()
    asyncio.run(run_load_test(args.concurrency, args.requests))
