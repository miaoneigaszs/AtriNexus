#!/usr/bin/env python3
"""
服务器时间诊断脚本
在服务器上运行此脚本检查时间配置
"""

import datetime
import time
import sys

try:
    import pytz
    HAS_PYTZ = True
except ImportError:
    HAS_PYTZ = False
    print("警告: pytz 未安装，请运行: pip install pytz")
    sys.exit(1)

def check_time():
    print("=" * 50)
    print("服务器时间诊断报告")
    print("=" * 50)
    
    # 1. 系统本地时间
    local_now = datetime.datetime.now()
    print(f"\n1. 系统本地时间: {local_now}")
    print(f"   系统本地时区: {time.tzname}")
    
    # 2. UTC时间
    utc_now = datetime.datetime.now(pytz.UTC)
    print(f"\n2. UTC时间: {utc_now.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 3. 北京时间
    beijing_tz = pytz.timezone('Asia/Shanghai')
    beijing_now = utc_now.astimezone(beijing_tz)
    print(f"\n3. 北京时间: {beijing_now.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 4. 使用原代码方式获取的时间（可能有问题）
    beijing_now_old = datetime.datetime.now(beijing_tz)
    print(f"\n4. 原代码方式获取的北京时间: {beijing_now_old.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 5. 时差计算
    print(f"\n5. 时差分析:")
    print(f"   UTC 与北京时间差: {(beijing_now.utcoffset().total_seconds() / 3600):.1f} 小时")
    print(f"   系统本地时间与UTC差: {time.timezone / -3600:.1f} 小时")
    
    # 6. 判断是否有问题
    print(f"\n6. 诊断结果:")
    hour_diff = abs((beijing_now.hour - local_now.hour)) % 24
    
    if hour_diff != 0 and hour_diff != 23:  # 允许1小时偏差（夏令时）
        print(f"   ⚠️ 警告: 系统本地时间与北京时间相差 {hour_diff} 小时!")
        print(f"   这很可能是时间错误的根本原因。")
    else:
        print(f"   ✅ 系统本地时间与北京时间一致")
    
    # 7. 建议
    print(f"\n7. 修复建议:")
    print(f"   如果系统时间不对，请运行:")
    print(f"   sudo timedatectl set-timezone Asia/Shanghai")
    print(f"   sudo ntpdate -u ntp.aliyun.com  # 同步时间")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    check_time()
