"""
主程序入口文件
负责启动 AtriNexus-WeCom 企业微信服务
读取环境变量 WECOM_HOST 和 WECOM_PORT，导入server.py中的start_server函数，启动FastAPI服务
"""

import os
import sys
import io

# 设置系统默认编码为 UTF-8
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 禁止生成__pycache__文件夹
sys.dont_write_bytecode = True

# 允许从任意工作目录执行 run.py 时仍能导入 src 和 data 包。
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


def main():
    """启动 AtriNexus-WeCom 服务"""
    print("=" * 50)
    print("  AtriNexus-WeCom 企业微信智能伙伴")
    print("=" * 50)
    print()

    # 确保必要目录存在
    required_dirs = ['data', 'logs', 'data/config', 'data/database']
    for dir_name in required_dirs:
        dir_path = os.path.join(root_dir, dir_name)
        os.makedirs(dir_path, exist_ok=True)

    # 启动 WeCom 服务
    from src.app.server import start_server

    host = os.environ.get('WECOM_HOST', '127.0.0.1')
    port = int(os.environ.get('WECOM_PORT', '8080'))

    print(f"  正在启动服务... http://{host}:{port}")
    print(f"  回调URL: https://<你的域名>/api/wechat/callback")
    print(f"  健康检查: http://{host}:{port}/health")
    print()

    start_server(host=host, port=port)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n正在关闭服务...")
        print("感谢使用 AtriNexus-WeCom，再见！")
    except Exception as e:
        print(f"服务启动失败: {str(e)}")
        sys.exit(1)
