"""
主程序入口文件
负责启动 AtriNexus-WeCom 企业微信服务
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

# 将项目根目录添加到Python路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

# 将src目录添加到Python路径
src_path = os.path.join(root_dir, 'src')
sys.path.append(src_path)

def main():
    """启动 AtriNexus-WeCom 服务"""
    print("=" * 50)
    print("  AtriNexus-WeCom 企业微信智能伙伴")
    print("=" * 50)
    print()

    # 清理缓存
    try:
        from src.utils.cleanup import cleanup_pycache
        cleanup_pycache()
    except Exception:
        pass

    # 确保必要目录存在
    required_dirs = ['data', 'logs', 'data/config', 'data/database']
    for dir_name in required_dirs:
        dir_path = os.path.join(root_dir, dir_name)
        os.makedirs(dir_path, exist_ok=True)

    # 启动 WeCom 服务
    from src.wecom.server import start_server

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
