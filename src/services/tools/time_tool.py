"""
时间查询工具

当 LLM 需要知道当前时间时调用，返回带时段、星期、农历的格式化时间字符串。
"""

import datetime
import logging

import pytz

from src.services.tools.base_tool import BaseTool

logger = logging.getLogger('wecom')


class TimeTool(BaseTool):
    """获取当前北京时间（含时段、星期、农历）"""

    @property
    def name(self) -> str:
        return "get_current_time"

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "获取当前的日期和时间。"
                    "当用户询问时间、日期，或需要知道当前时间来回答问题时调用此工具。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }

    def execute(self, **kwargs) -> str:
        """返回格式化的当前北京时间"""
        tz = pytz.timezone('Asia/Shanghai')
        now = datetime.datetime.now(pytz.UTC).astimezone(tz)

        weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        weekday = weekdays[now.weekday()]
        hour = now.hour

        if 5 <= hour < 12:
            period = "上午"
        elif 12 <= hour < 14:
            period = "中午"
        elif 14 <= hour < 18:
            period = "下午"
        elif 18 <= hour < 22:
            period = "晚上"
        else:
            period = "深夜"

        lunar_str = ""
        try:
            from zhdate import ZhDate
            lunar_date = ZhDate.from_datetime(now)
            lunar_str = f"，农历{lunar_date.chinese()}"
        except Exception:
            pass

        result = (
            f"当前时间是{now.strftime('%Y年%m月%d日')} {weekday} "
            f"{now.strftime('%H:%M:%S')}（{period}）{lunar_str}"
        )
        logger.info(f"[TimeTool] {result}")
        return result
