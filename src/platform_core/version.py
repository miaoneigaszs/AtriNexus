
import json
import os

def get_current_version() -> str:
    """获取当前版本号"""
    try:
        # 假设 version.json 在项目根目录
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        version_file = os.path.join(root_dir, 'version.json')
        if os.path.exists(version_file):
            with open(version_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('version', 'unknown')
    except Exception:
        pass
    return 'unknown'

def get_version_identifier() -> str:
    """获取版本标识符"""
    return f"AtriNexus/{get_current_version()}"
