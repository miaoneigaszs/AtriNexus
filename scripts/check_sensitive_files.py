#!/usr/bin/env python3
"""
开源前敏感文件检查脚本
用于验证 .gitignore 配置是否正确，确保敏感文件不会被上传到 GitHub
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple


class Color:
    """终端颜色"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """打印标题"""
    print(f"\n{Color.BOLD}{Color.CYAN}{'='*60}{Color.RESET}")
    print(f"{Color.BOLD}{Color.CYAN}{text.center(60)}{Color.RESET}")
    print(f"{Color.BOLD}{Color.CYAN}{'='*60}{Color.RESET}\n")


def print_success(text: str):
    """打印成功信息"""
    print(f"{Color.GREEN}✓ {text}{Color.RESET}")


def print_error(text: str):
    """打印错误信息"""
    print(f"{Color.RED}✗ {text}{Color.RESET}")


def print_warning(text: str):
    """打印警告信息"""
    print(f"{Color.YELLOW}⚠ {text}{Color.RESET}")


def print_info(text: str):
    """打印信息"""
    print(f"{Color.BLUE}ℹ {text}{Color.RESET}")


def check_file_exists(filepath: str) -> bool:
    """检查文件是否存在"""
    return os.path.exists(filepath)


def check_sensitive_keywords(directory: str) -> Dict[str, List[str]]:
    """检查提交列表中的敏感关键词"""
    # 这里简化处理，实际应该解析 git status
    sensitive_patterns = {
        'api_key': [],
        'secret': [],
        'password': [],
        'token': [],
        'sk-': []
    }
    
    # 检查 config.json.template
    template_file = os.path.join(directory, 'data/config/config.json.template')
    if os.path.exists(template_file):
        try:
            with open(template_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # 检查是否有硬编码的密钥
                if 'sk-' in content and '"value": "sk-' in content:
                    sensitive_patterns['sk-'].append(template_file)
        except Exception as e:
            print_warning(f"无法读取 {template_file}: {e}")
    
    return sensitive_patterns


def check_critical_files(directory: str) -> Tuple[List[str], List[str]]:
    """检查关键文件"""
    print_header("检查关键文件")
    
    must_ignore = [
        'data/config/config.json',
        'data/config/backups',
        'logs',
        '.env',
    ]
    
    must_include = [
        'config.json.template',
        'README.md',
        'LICENSE',
        '.gitignore',
        'requirements.txt',
    ]
    
    ignored_ok = []
    ignored_fail = []
    
    # 检查必须忽略的文件
    print_info("检查必须忽略的敏感文件:")
    for file in must_ignore:
        filepath = os.path.join(directory, file)
        # 这里简化，实际应该检查 git status
        if check_file_exists(filepath):
            print_warning(f"文件存在（确保已在 .gitignore 中）: {file}")
            ignored_ok.append(file)
        else:
            print_info(f"文件不存在: {file}")
    
    # 检查必须包含的文件
    print_info("\n检查必须包含的重要文件:")
    included_ok = []
    included_fail = []
    for file in must_include:
        filepath = os.path.join(directory, file)
        if check_file_exists(filepath):
            print_success(f"文件存在: {file}")
            included_ok.append(file)
        else:
            print_error(f"文件缺失: {file}")
            included_fail.append(file)
    
    return included_ok, included_fail


def check_database_files(directory: str) -> List[str]:
    """检查数据库文件"""
    print_header("检查数据库文件")
    
    db_patterns = ['*.db', '*.sqlite', '*.sqlite3']
    db_files = []
    
    # 搜索数据库文件
    for root, dirs, files in os.walk(directory):
        # 跳过 .git 目录
        if '.git' in root:
            continue
        
        for file in files:
            if any(file.endswith(ext.replace('*', '')) for ext in db_patterns):
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, directory)
                db_files.append(rel_path)
    
    if db_files:
        print_warning("发现数据库文件（确保已在 .gitignore 中）:")
        for file in db_files:
            print_warning(f"  - {file}")
    else:
        print_success("未发现数据库文件")
    
    return db_files


def check_data_directories(directory: str) -> Dict[str, bool]:
    """检查数据目录"""
    print_header("检查敏感数据目录")
    
    sensitive_dirs = {
        'data/database': True,      # 必须忽略
        'data/vectordb': True,      # 必须忽略
        'data/knowledge': True,     # 必须忽略
        'data/token_stats': True,   # 必须忽略
        'data/avatars/ATRI': False, # 必须包含
        'data/avatars/MONO': False, # 必须包含
        'data/avatars/Nijiko': False, # 必须包含
    }
    
    results = {}
    for dir_path, should_ignore in sensitive_dirs.items():
        full_path = os.path.join(directory, dir_path)
        exists = os.path.exists(full_path)
        
        if should_ignore:
            if exists:
                print_warning(f"目录存在（应忽略）: {dir_path}")
            else:
                print_info(f"目录不存在: {dir_path}")
        else:
            if exists:
                print_success(f"必需目录存在: {dir_path}")
            else:
                print_error(f"必需目录缺失: {dir_path}")
        
        results[dir_path] = exists
    
    return results


def check_gitignore_config(directory: str) -> bool:
    """检查 .gitignore 配置"""
    print_header("检查 .gitignore 配置")
    
    gitignore_file = os.path.join(directory, '.gitignore')
    
    if not os.path.exists(gitignore_file):
        print_error(".gitignore 文件不存在！")
        return False
    
    print_success(".gitignore 文件存在")
    
    # 检查关键配置
    must_have_patterns = [
        'config.json',
        '*.db',
        '*.sqlite',
        'logs/',
        '.env',
    ]
    
    try:
        with open(gitignore_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print_info("\n检查关键忽略规则:")
        all_ok = True
        for pattern in must_have_patterns:
            if pattern in content:
                print_success(f"已配置: {pattern}")
            else:
                print_error(f"缺失规则: {pattern}")
                all_ok = False
        
        # 检查是否包含新的敏感目录
        new_patterns = [
            'data/database/',
            'data/vectordb/',
            'data/knowledge/',
        ]
        
        print_info("\n检查新增的敏感目录规则:")
        for pattern in new_patterns:
            if pattern in content:
                print_success(f"已配置: {pattern}")
            else:
                print_warning(f"建议添加: {pattern}")
        
        return all_ok
    
    except Exception as e:
        print_error(f"读取 .gitignore 失败: {e}")
        return False


def generate_report(directory: str) -> Dict:
    """生成检查报告"""
    print_header("生成检查报告")
    
    report = {
        'project_dir': directory,
        'checks': {}
    }
    
    # 执行各项检查
    report['checks']['gitignore'] = check_gitignore_config(directory)
    report['checks']['critical_files'] = check_critical_files(directory)
    report['checks']['database_files'] = check_database_files(directory)
    report['checks']['data_directories'] = check_data_directories(directory)
    
    return report


def print_summary(report: Dict):
    """打印总结"""
    print_header("检查总结")
    
    checks = report.get('checks', {})
    
    # 统计
    total = len(checks)
    passed = sum(1 for v in checks.values() if isinstance(v, bool) and v)
    
    print_info(f"项目目录: {report['project_dir']}")
    print_info(f"检查项目: {total}")
    
    if passed == total:
        print_success(f"通过检查: {passed}/{total}")
        print_success("\n✅ 项目已准备好开源！")
    else:
        print_warning(f"通过检查: {passed}/{total}")
        print_warning("\n⚠️  请修复上述问题后再开源！")
    
    print("\n" + "="*60)


def main():
    """主函数"""
    # 获取项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    print(f"\n{Color.BOLD}{Color.MAGENTA}AtriNexus 开源前检查工具{Color.RESET}")
    print(f"{Color.BOLD}{Color.MAGENTA}检查目录: {project_dir}{Color.RESET}\n")
    
    # 生成报告
    report = generate_report(project_dir)
    
    # 打印总结
    print_summary(report)
    
    # 提示后续操作
    print_header("后续操作建议")
    print_info("1. 确认所有敏感文件已添加到 .gitignore")
    print_info("2. 使用 'git status' 查看将要提交的文件")
    print_info("3. 搜索代码中的硬编码密钥")
    print_info("4. 检查 README.md 是否包含完整的配置说明")
    print_info("5. 确认 LICENSE 文件存在并选择合适的协议")
    print()
    print_info("参考文档: 开源部署指南.md")
    print()


if __name__ == '__main__':
    main()
