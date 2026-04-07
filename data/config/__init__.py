import os
import json
import logging
import shutil
import difflib
import re
from dataclasses import dataclass
from typing import List, Dict, Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()


SECRET_ENV_MAP = {
    "llm.api_key": "ATRINEXUS_LLM_API_KEY",
    "media.image_recognition.api_key": "ATRINEXUS_VISION_API_KEY",
    "network_search.api_key": "ATRINEXUS_NETWORK_SEARCH_API_KEY",
    "intent_recognition.api_key": "ATRINEXUS_INTENT_API_KEY",
    "embedding.api_key": "ATRINEXUS_EMBEDDING_API_KEY",
    "wecom.secret": "ATRINEXUS_WECOM_SECRET",
    "wecom.token": "ATRINEXUS_WECOM_TOKEN",
    "wecom.encoding_aes_key": "ATRINEXUS_WECOM_ENCODING_AES_KEY",
    "auth.admin_password": "ATRINEXUS_ADMIN_PASSWORD",
}


def get_env_override(key: str, fallback: str = "") -> str:
    """Read a sensitive setting from environment first, with config file fallback."""
    env_name = SECRET_ENV_MAP.get(key)
    if not env_name:
        return fallback

    env_value = os.getenv(env_name)
    if env_value is None:
        return fallback

    value = env_value.strip()
    if not value:
        return fallback

    logger.debug("敏感配置已由环境变量覆盖: %s", env_name)
    return value


def validate_api_key(api_key: str, key_name: str = "API Key") -> bool:
    """
    验证 API Key 格式是否有效
    
    Args:
        api_key: 待验证的 API Key
        key_name: 用于日志的键名
        
    Returns:
        bool: True 如果有效，False 如果无效
    """
    if not api_key:
        logger.warning(f"{key_name} 为空")
        return False
    
    # 检测中文占位符
    if re.search(r'[\u4e00-\u9fff]', api_key):
        logger.error(
            f"❌ {key_name} 包含中文字符，可能是占位符！"
            f"请检查配置文件 data/config/config.json 中的 {key_name}"
        )
        return False
    
    # 检测常见占位符模式
    placeholder_patterns = [
        r'你的',
        r'替换为',
        r'请填写',
        r'请输入',
        r'your_',
        r'replace_',
        r'<.*?>',
        r'xxx+',
    ]
    for pattern in placeholder_patterns:
        if re.search(pattern, api_key, re.IGNORECASE):
            logger.error(
                f"❌ {key_name} 看起来是占位符（匹配 '{pattern}'）"
                f"请检查配置文件 data/config/config.json"
            )
            return False
    
    # 检查是否过短
    if len(api_key) < 8:
        logger.warning(f"{key_name} 长度过短({len(api_key)}字符)，可能无效")
        return False
    
    return True

@dataclass
class GroupChatConfigItem:
    id: str
    group_name: str
    avatar: str
    triggers: List[str]
    enable_at_trigger: bool = True  # 默认启用@触发

@dataclass
class UserSettings:
    listen_list: List[str]
    group_chat_config: List[GroupChatConfigItem] = None
    
    def __post_init__(self):
        if self.group_chat_config is None:
            self.group_chat_config = []

@dataclass
class LLMSettings:
    api_key: str
    base_url: str
    model: str
    max_tokens: int
    temperature: float
    auto_model_switch: bool = False
    fallback_models: List[str] = None

    def __post_init__(self):
        if self.fallback_models is None:
            self.fallback_models = []

@dataclass
class ImageRecognitionSettings:
    api_key: str
    base_url: str
    temperature: float
    model: str

@dataclass
class MediaSettings:
    image_recognition: ImageRecognitionSettings

@dataclass
class AutoMessageSettings:
    content: str
    min_hours: float
    max_hours: float

@dataclass
class QuietTimeSettings:
    start: str
    end: str

@dataclass
class ContextSettings:
    max_groups: int
    avatar_dir: str  # 人设目录路径，prompt文件和表情包目录都将基于此路径

@dataclass
class MessageQueueSettings:
    timeout: int

@dataclass
class TaskSettings:
    task_id: str
    chat_id: str
    content: str
    schedule_type: str
    schedule_time: str
    is_active: bool

@dataclass
class ScheduleSettings:
    tasks: List[TaskSettings]

@dataclass
class BehaviorSettings:
    auto_message: AutoMessageSettings
    quiet_time: QuietTimeSettings
    context: ContextSettings
    schedule_settings: ScheduleSettings
    message_queue: MessageQueueSettings

@dataclass
class AuthSettings:
    admin_password: str

@dataclass
class NetworkSearchSettings:
    search_enabled: bool
    weblens_enabled: bool
    api_key: str
    base_url: str
    search_provider: str = 'kourichat'

@dataclass
class IntentRecognitionSettings:
    api_key: str
    base_url: str
    model: str
    temperature: float


@dataclass
class EmbeddingSettings:
    """Embedding 服务配置"""
    api_key: str
    base_url: str
    model: str
    reranker_model: str


@dataclass
class WeComSettings:
    corp_id: str
    agent_id: str
    secret: str
    token: str
    encoding_aes_key: str


@dataclass
class ThreadPoolSettings:
    max_workers: int


@dataclass
class HTTPClientSettings:
    timeout: float
    connect_timeout: float
    max_connections: int
    max_keepalive_connections: int
    keepalive_expiry: float


@dataclass
class SystemPerformanceSettings:
    thread_pool: ThreadPoolSettings
    http_client: HTTPClientSettings


@dataclass
class FileUploadSettings:
    max_size_mb: int
    allowed_types: List[str]


@dataclass
class KnowledgeBaseSettings:
    file_upload: FileUploadSettings


@dataclass
class CompanionModeSettings:
    triggers: List[str]

@dataclass
class Config:
    def __init__(self, auto_migrate: bool = False):
        self.user: UserSettings
        self.llm: LLMSettings
        self.media: MediaSettings
        self.behavior: BehaviorSettings
        self.auth: AuthSettings
        self.network_search: NetworkSearchSettings
        self.intent_recognition: IntentRecognitionSettings
        self.embedding: EmbeddingSettings  # 新增
        self.wecom: WeComSettings
        self.system_performance: SystemPerformanceSettings
        self.kb: KnowledgeBaseSettings
        self.companion_mode: CompanionModeSettings
        self.version: str = "1.0.0"  # 配置文件版本
        self.auto_migrate = auto_migrate
        self.load_config()

    @property
    def config_dir(self) -> str:
        """返回配置文件所在目录"""
        return os.path.dirname(__file__)

    @property
    def config_path(self) -> str:
        """返回配置文件完整路径"""
        return os.path.join(self.config_dir, 'config.json')

    @property
    def config_template_path(self) -> str:
        """返回配置模板文件完整路径"""
        return os.path.join(self.config_dir, 'config.json.template')

    @property
    def config_template_bak_path(self) -> str:
        """返回备份的配置模板文件完整路径"""
        return os.path.join(self.config_dir, 'config.json.template.bak')

    @property
    def config_backup_dir(self) -> str:
        """返回配置备份目录路径"""
        backup_dir = os.path.join(self.config_dir, 'backups')
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
        return backup_dir

    def backup_config(self) -> str:
        """备份当前配置文件，仅在配置发生变更时进行备份，并覆盖之前的备份

        Returns:
            str: 备份文件路径
        """
        if not os.path.exists(self.config_path):
            logger.warning("无法备份配置文件：文件不存在")
            return ""

        backup_filename = "config_backup.json"
        backup_path = os.path.join(self.config_backup_dir, backup_filename)

        # 检查是否需要备份
        if os.path.exists(backup_path):
            # 比较当前配置文件和备份文件的内容
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f1, \
                     open(backup_path, 'r', encoding='utf-8') as f2:
                    if f1.read() == f2.read():
                        # 内容相同，无需备份
                        logger.debug("配置未发生变更，跳过备份")
                        return backup_path
            except Exception as e:
                logger.error(f"比较配置文件失败: {str(e)}")

        try:
            # 内容不同或备份不存在，进行备份
            shutil.copy2(self.config_path, backup_path)
            logger.info(f"已备份配置文件到: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"备份配置文件失败: {str(e)}")
            return ""

    def _read_json_file(self, path: str) -> dict:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _resolve_readable_config_path(self) -> str:
        """返回当前可读取的配置来源。

        生产环境允许只有模板文件存在，密钥由环境变量覆盖；
        只有在需要持久化用户修改时，才要求物化 config.json。
        """
        if os.path.exists(self.config_path):
            return self.config_path
        if os.path.exists(self.config_template_path):
            return self.config_template_path
        raise FileNotFoundError("配置文件和模板文件都不存在")

    def _ensure_config_file_exists(self) -> None:
        """在需要写配置时，确保 config.json 物理存在。"""
        if os.path.exists(self.config_path):
            return
        if not os.path.exists(self.config_template_path):
            raise FileNotFoundError("无法创建配置文件：模板文件不存在")
        logger.info("配置文件不存在，基于模板创建可写配置文件")
        shutil.copy2(self.config_template_path, self.config_path)
        self._backup_template()

    def _backup_template(self, force=False):
        # 如果模板备份不存在或强制备份，创建备份
        if force or not os.path.exists(self.config_template_bak_path):
            try:
                shutil.copy2(self.config_template_path, self.config_template_bak_path)
                logger.info(f"已创建模板配置备份: {self.config_template_bak_path}")
                return True
            except Exception as e:
                logger.warning(f"创建模板配置备份失败: {str(e)}")
                return False
        return False

    def compare_configs(self, old_config: Dict[str, Any], new_config: Dict[str, Any], path: str = "") -> Dict[str, Any]:
        # 比较两个配置字典的差异
        diff = {"added": {}, "removed": {}, "modified": {}}

        # 检查添加和修改的字段
        for key, new_value in new_config.items():
            current_path = f"{path}.{key}" if path else key

            if key not in old_config:
                # 新增字段
                diff["added"][current_path] = new_value
            elif isinstance(new_value, dict) and isinstance(old_config[key], dict):
                # 递归比较子字典
                sub_diff = self.compare_configs(old_config[key], new_value, current_path)
                # 合并子字典的差异
                for diff_type in ["added", "removed", "modified"]:
                    diff[diff_type].update(sub_diff[diff_type])
            elif new_value != old_config[key]:
                # 修改的字段
                diff["modified"][current_path] = {"old": old_config[key], "new": new_value}

        # 检查删除的字段
        for key in old_config:
            current_path = f"{path}.{key}" if path else key
            if key not in new_config:
                diff["removed"][current_path] = old_config[key]

        return diff

    def generate_diff_report(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> str:
        # 生成配置差异报告
        old_json = json.dumps(old_config, indent=4, ensure_ascii=False).splitlines()
        new_json = json.dumps(new_config, indent=4, ensure_ascii=False).splitlines()
        diff = difflib.unified_diff(old_json, new_json, fromfile='old_config', tofile='new_config', lineterm='')
        return '\n'.join(diff)

    def merge_configs(self, current: dict, template: dict, old_template: dict = None) -> dict:
        # 智能合并配置
        result = current.copy()
        for key, value in template.items():
            # 新字段或非字典字段
            if key not in current:
                result[key] = value
            # 字典字段需要递归合并
            elif isinstance(value, dict) and isinstance(current[key], dict):
                old_value = old_template.get(key, {}) if old_template else None
                result[key] = self.merge_configs(current[key], value, old_value)
            # 如果用户值与旧模板相同，但新模板已更新，则使用新值
            elif old_template and key in old_template and current[key] == old_template[key] and value != old_template[key]:
                logger.debug(f"字段 '{key}' 更新为新模板值")
                result[key] = value
        return result

    def save_config(self, config_data: dict) -> bool:
        # 保存配置到文件
        try:
            self._ensure_config_file_exists()
            # 备份当前配置
            self.backup_config()

            # 读取现有配置
            current_config = self._read_json_file(self.config_path)

            # 合并新配置
            for key, value in config_data.items():
                if key in current_config and isinstance(current_config[key], dict) and isinstance(value, dict):
                    self._recursive_update(current_config[key], value)
                else:
                    current_config[key] = value

            # 保存更新后的配置
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(current_config, f, indent=4, ensure_ascii=False)

            return True
        except Exception as e:
            logger.error(f"保存配置失败: {str(e)}")
            return False

    def _recursive_update(self, target: dict, source: dict):
        # 递归更新字典
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._recursive_update(target[key], value)
            else:
                target[key] = value

    def _check_and_update_config(self) -> None:
        # 检查并更新配置文件
        try:
            # 检查模板文件是否存在
            if not os.path.exists(self.config_template_path):
                logger.warning(f"模板配置文件不存在: {self.config_template_path}")
                return

            self._ensure_config_file_exists()
            # 读取配置文件
            current_config = self._read_json_file(self.config_path)

            template_config = self._read_json_file(self.config_template_path)

            # 创建备份模板
            self._backup_template()

            # 读取备份模板
            old_template_config = None
            if os.path.exists(self.config_template_bak_path):
                try:
                    with open(self.config_template_bak_path, 'r', encoding='utf-8') as f:
                        old_template_config = json.load(f)
                except Exception as e:
                    logger.warning(f"读取备份模板失败: {str(e)}")

            # 比较配置差异
            diff = self.compare_configs(current_config, template_config)

            # 如果有差异，更新配置
            if any(diff.values()):
                logger.info("检测到配置需要更新")

                # 备份当前配置
                backup_path = self.backup_config()
                if backup_path:
                    logger.info(f"已备份原配置到: {backup_path}")

                # 合并配置
                updated_config = self.merge_configs(current_config, template_config, old_template_config)

                # 保存更新后的配置
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(updated_config, f, indent=4, ensure_ascii=False)

                logger.info("配置文件已更新")
            else:
                logger.debug("配置文件无需更新")

        except Exception as e:
            logger.error(f"检查配置更新失败: {str(e)}")
            raise

    def load_config(self, auto_migrate: bool | None = None) -> None:
        # 加载配置文件
        try:
            if auto_migrate is None:
                auto_migrate = self.auto_migrate

            # 配置迁移会写入磁盘，只在显式启用时执行
            if auto_migrate:
                self._check_and_update_config()

            config_source = self._resolve_readable_config_path()
            if config_source == self.config_template_path:
                logger.info("未检测到 config.json，使用配置模板并通过环境变量覆盖敏感字段")

            # 读取配置文件
            with open(config_source, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                categories = config_data['categories']

                # 用户设置（兼容旧版：user_settings 已从 config.json 移除）
                user_data = categories.get('user_settings', {}).get('settings', {})
                listen_list = user_data.get('listen_list', {}).get('value', [])
                # 确保listen_list是列表类型
                if not isinstance(listen_list, list):
                    listen_list = [str(listen_list)] if listen_list else []
                
                # 群聊配置
                group_chat_config_data = user_data.get('group_chat_config', {}).get('value', [])
                group_chat_configs = []
                if isinstance(group_chat_config_data, list):
                    for config_item in group_chat_config_data:
                        if isinstance(config_item, dict) and all(key in config_item for key in ['id', 'groupName', 'avatar', 'triggers']):
                            group_chat_configs.append(GroupChatConfigItem(
                                id=config_item['id'],
                                group_name=config_item['groupName'],
                                avatar=config_item['avatar'],
                                triggers=config_item.get('triggers', []),
                                enable_at_trigger=config_item.get('enableAtTrigger', True)  # 默认启用@触发
                            ))
                
                self.user = UserSettings(
                    listen_list=listen_list,
                    group_chat_config=group_chat_configs
                )

                # LLM设置
                llm_data = categories.get('llm_settings', {}).get('settings', {})
                # fallback_models: 备用模型列表
                fallback_models_data = llm_data.get('fallback_models', {}).get('value', [])
                if not isinstance(fallback_models_data, list):
                    fallback_models_data = []

                self.llm = LLMSettings(
                    api_key=get_env_override('llm.api_key', llm_data.get('api_key', {}).get('value', '')),
                    base_url=llm_data.get('base_url', {}).get('value', ''),
                    model=llm_data.get('model', {}).get('value', ''),
                    max_tokens=int(llm_data.get('max_tokens', {}).get('value', 0)),
                    temperature=float(llm_data.get('temperature', {}).get('value', 0)),
                    auto_model_switch=bool(llm_data.get('auto_model_switch', {}).get('value', False)),
                    fallback_models=fallback_models_data
                )

                # 媒体设置
                media_data = categories.get('media_settings', {}).get('settings', {})
                image_recognition_data = media_data.get('image_recognition', {})

                self.media = MediaSettings(
                    image_recognition=ImageRecognitionSettings(
                        api_key=get_env_override(
                            'media.image_recognition.api_key',
                            image_recognition_data.get('api_key', {}).get('value', ''),
                        ),
                        base_url=image_recognition_data.get('base_url', {}).get('value', ''),
                        temperature=float(image_recognition_data.get('temperature', {}).get('value', 0)),
                        model=image_recognition_data.get('model', {}).get('value', '')
                    )
                )

                # 行为设置
                behavior_data = categories.get('behavior_settings', {}).get('settings', {})
                auto_message_data = behavior_data.get('auto_message', {})
                auto_message_countdown = auto_message_data.get('countdown', {})
                quiet_time_data = behavior_data.get('quiet_time', {})
                context_data = behavior_data.get('context', {})

                # 消息队列设置
                message_queue_data = behavior_data.get('message_queue', {})
                message_queue_timeout = message_queue_data.get('timeout', {}).get('value', 8)

                # 确保目录路径规范化
                avatar_dir = context_data.get('avatar_dir', {}).get('value', '')
                if avatar_dir and not avatar_dir.startswith('data/avatars/'):
                    avatar_dir = f"data/avatars/{avatar_dir.split('/')[-1]}"

                # 定时任务配置
                schedule_tasks = []
                if 'schedule_settings' in categories:
                    schedule_data = categories['schedule_settings']
                    if 'settings' in schedule_data and 'tasks' in schedule_data['settings']:
                        tasks_data = schedule_data['settings']['tasks'].get('value', [])
                        for task in tasks_data:
                            # 确保必要的字段存在
                            if all(key in task for key in ['task_id', 'chat_id', 'content', 'schedule_type', 'schedule_time']):
                                schedule_tasks.append(TaskSettings(
                                    task_id=task['task_id'],
                                    chat_id=task['chat_id'],
                                    content=task['content'],
                                    schedule_type=task['schedule_type'],
                                    schedule_time=task['schedule_time'],
                                    is_active=task.get('is_active', True)
                                ))

                # 行为配置
                self.behavior = BehaviorSettings(
                    auto_message=AutoMessageSettings(
                        content=auto_message_data.get('content', {}).get('value', ''),
                        min_hours=float(auto_message_countdown.get('min_hours', {}).get('value', 0)),
                        max_hours=float(auto_message_countdown.get('max_hours', {}).get('value', 0))
                    ),
                    quiet_time=QuietTimeSettings(
                        start=quiet_time_data.get('start', {}).get('value', ''),
                        end=quiet_time_data.get('end', {}).get('value', '')
                    ),
                    context=ContextSettings(
                        max_groups=int(context_data.get('max_groups', {}).get('value', 0)),
                        avatar_dir=avatar_dir
                    ),
                    schedule_settings=ScheduleSettings(
                        tasks=schedule_tasks
                    ),
                    message_queue=MessageQueueSettings(
                        timeout=int(message_queue_timeout)
                    )
                )

                # 认证设置
                auth_data = categories.get('auth_settings', {}).get('settings', {})
                self.auth = AuthSettings(
                    admin_password=get_env_override(
                        'auth.admin_password',
                        auth_data.get('admin_password', {}).get('value', ''),
                    )
                )

                # 网络搜索设置（使用安全的默认值）
                try:
                    network_search_data = categories.get('network_search_settings', {}).get('settings', {})
                    self.network_search = NetworkSearchSettings(
                        search_enabled=network_search_data.get('search_enabled', {}).get('value', False),
                        weblens_enabled=network_search_data.get('weblens_enabled', {}).get('value', False),
                        api_key=get_env_override(
                            'network_search.api_key',
                            network_search_data.get('api_key', {}).get('value', ''),
                        ),
                        base_url=network_search_data.get('base_url', {}).get('value', 'https://api.kourichat.com/v1'),
                        search_provider=network_search_data.get('search_provider', {}).get('value', 'kourichat')
                    )
                except Exception as e:
                    logger.warning(f"加载网络搜索设置失败，使用默认值: {e}")
                    self.network_search = NetworkSearchSettings(
                        search_enabled=False,
                        weblens_enabled=False,
                        api_key='',
                        base_url='https://api.kourichat.com/v1',
                        search_provider='kourichat'
                    )

                # 意图识别设置（使用安全的默认值）
                try:
                    intent_recognition_data = categories.get('intent_recognition_settings', {}).get('settings', {})
                    self.intent_recognition = IntentRecognitionSettings(
                        api_key=get_env_override(
                            'intent_recognition.api_key',
                            intent_recognition_data.get('api_key', {}).get('value', ''),
                        ),
                        base_url=intent_recognition_data.get('base_url', {}).get('value', 'https://api.siliconflow.cn/v1'),
                        model=intent_recognition_data.get('model', {}).get('value', 'Qwen/Qwen2.5-7B-Instruct'),
                        temperature=float(intent_recognition_data.get('temperature', {}).get('value', 0.1))
                    )
                except Exception as e:
                    logger.warning(f"加载意图识别设置失败，使用默认值: {e}")
                    self.intent_recognition = IntentRecognitionSettings(
                        api_key='',
                        base_url='https://api.siliconflow.cn/v1',
                        model='Qwen/Qwen2.5-7B-Instruct',
                        temperature=0.1
                    )

                # Embedding 设置（使用安全的默认值）
                try:
                    embedding_data = categories.get('embedding_settings', {}).get('settings', {})
                    self.embedding = EmbeddingSettings(
                        api_key=get_env_override(
                            'embedding.api_key',
                            embedding_data.get('api_key', {}).get('value', ''),
                        ),
                        base_url=embedding_data.get('base_url', {}).get('value', 'https://api.siliconflow.cn/v1'),
                        model=embedding_data.get('model', {}).get('value', 'BAAI/bge-m3'),
                        reranker_model=embedding_data.get('reranker_model', {}).get('value', 'BAAI/bge-reranker-v2-m3')
                    )
                    logger.info(f"Embedding配置加载完成: model={self.embedding.model}, base_url={self.embedding.base_url}")
                except Exception as e:
                    logger.warning(f"加载 Embedding 设置失败，使用默认值: {e}")
                    self.embedding = EmbeddingSettings(
                        api_key='',
                        base_url='https://api.siliconflow.cn/v1',
                        model='BAAI/bge-m3',
                        reranker_model='BAAI/bge-reranker-v2-m3'
                    )

                # 企业微信设置
                wecom_data = categories.get('wecom_settings', {}).get('settings', {})
                self.wecom = WeComSettings(
                    corp_id=wecom_data.get('corp_id', {}).get('value', ''),
                    agent_id=wecom_data.get('agent_id', {}).get('value', ''),
                    secret=get_env_override('wecom.secret', wecom_data.get('secret', {}).get('value', '')),
                    token=get_env_override('wecom.token', wecom_data.get('token', {}).get('value', '')),
                    encoding_aes_key=get_env_override(
                        'wecom.encoding_aes_key',
                        wecom_data.get('encoding_aes_key', {}).get('value', ''),
                    ),
                )

                # 陪伴模式设置
                companion_triggers = categories.get('behavior_settings', {}).get('settings', {}).get('companion_mode_trigger', {}).get('value', ['ATRI,在吗', '聊聊天吧'])
                if not isinstance(companion_triggers, list):
                    companion_triggers = [str(companion_triggers)] if companion_triggers else []
                self.companion_mode = CompanionModeSettings(
                    triggers=companion_triggers
                )

                # 系统性能设置（使用安全的默认值）
                try:
                    system_performance_data = categories.get('system_performance_settings', {}).get('settings', {})
                    thread_pool_data = system_performance_data.get('thread_pool', {})
                    http_client_data = system_performance_data.get('http_client', {})

                    self.system_performance = SystemPerformanceSettings(
                        thread_pool=ThreadPoolSettings(
                            max_workers=int(thread_pool_data.get('max_workers', {}).get('value', 4))
                        ),
                        http_client=HTTPClientSettings(
                            timeout=float(http_client_data.get('timeout', {}).get('value', 60.0)),
                            connect_timeout=float(http_client_data.get('connect_timeout', {}).get('value', 10.0)),
                            max_connections=int(http_client_data.get('max_connections', {}).get('value', 20)),
                            max_keepalive_connections=int(http_client_data.get('max_keepalive_connections', {}).get('value', 10)),
                            keepalive_expiry=float(http_client_data.get('keepalive_expiry', {}).get('value', 30.0))
                        )
                    )
                except Exception as e:
                    logger.warning(f"加载系统性能设置失败，使用默认值: {e}")
                    self.system_performance = SystemPerformanceSettings(
                        thread_pool=ThreadPoolSettings(max_workers=4),
                        http_client=HTTPClientSettings(
                            timeout=60.0,
                            connect_timeout=10.0,
                            max_connections=20,
                            max_keepalive_connections=10,
                            keepalive_expiry=30.0
                        )
                    )

                # 知识库设置（使用安全的默认值）
                try:
                    kb_data = categories.get('knowledge_base_settings', {}).get('settings', {})
                    file_upload_data = kb_data.get('file_upload', {})
                    self.kb = KnowledgeBaseSettings(
                        file_upload=FileUploadSettings(
                            max_size_mb=int(file_upload_data.get('max_size_mb', {}).get('value', 20)),
                            allowed_types=file_upload_data.get('allowed_types', {}).get('value', [])
                        )
                    )
                except Exception as e:
                    logger.warning(f"加载知识库设置失败，使用默认值: {e}")
                    self.kb = KnowledgeBaseSettings(
                        file_upload=FileUploadSettings(
                            max_size_mb=20,
                            allowed_types=['.pdf', '.docx', '.doc', '.txt', '.md']
                        )
                    )

                logger.info("配置加载完成")
                
                # 验证关键 API Key 配置
                self._validate_api_keys()

        except Exception as e:
            logger.error(f"加载配置失败: {str(e)}")
            raise
    
    def _validate_api_keys(self):
        """验证关键 API Key 配置是否有效"""
        api_keys_to_check = [
            (self.llm.api_key, "LLM API Key"),
            (self.intent_recognition.api_key, "意图识别 API Key") if self.intent_recognition.api_key else None,
            (self.network_search.api_key, "网络搜索 API Key") if self.network_search.api_key else None,
        ]
        
        has_invalid_key = False
        for key_info in api_keys_to_check:
            if key_info:
                api_key, key_name = key_info
                if api_key and not validate_api_key(api_key, key_name):
                    has_invalid_key = True
        
        if has_invalid_key:
            logger.warning(
                "⚠️ 检测到无效的 API Key 配置！服务可能无法正常工作。"
                "请检查环境变量覆盖或 data/config/config.json 中的配置。"
            )

    # 更新管理员密码
    def update_password(self, password: str) -> bool:
        try:
            config_data = {
                'categories': {
                    'auth_settings': {
                        'settings': {
                            'admin_password': {
                                'value': password
                            }
                        }
                    }
                }
            }
            return self.save_config(config_data)
        except Exception as e:
            logger.error(f"更新密码失败: {str(e)}")
            return False

# 创建全局配置实例
config = Config(auto_migrate=os.getenv("ATRINEXUS_AUTO_MIGRATE_CONFIG", "").lower() in {"1", "true", "yes", "on"})

# 为了兼容性保留的旧变量（将在未来版本中移除）
LISTEN_LIST = config.user.listen_list
DEEPSEEK_API_KEY = config.llm.api_key
DEEPSEEK_BASE_URL = config.llm.base_url
MODEL = config.llm.model
MAX_TOKEN = config.llm.max_tokens
TEMPERATURE = config.llm.temperature
VISION_API_KEY = config.media.image_recognition.api_key
VISION_BASE_URL = config.media.image_recognition.base_url
VISION_TEMPERATURE = config.media.image_recognition.temperature
MAX_GROUPS = config.behavior.context.max_groups
AUTO_MESSAGE = config.behavior.auto_message.content
MIN_COUNTDOWN_HOURS = config.behavior.auto_message.min_hours
MAX_COUNTDOWN_HOURS = config.behavior.auto_message.max_hours
QUIET_TIME_START = config.behavior.quiet_time.start
QUIET_TIME_END = config.behavior.quiet_time.end

# 网络搜索设置
NETWORK_SEARCH_ENABLED = config.network_search.search_enabled
NETWORK_SEARCH_MODEL = 'kourichat-search'  # 固定使用AtriNexus模型
WEBLENS_ENABLED = config.network_search.weblens_enabled
WEBLENS_MODEL = 'kourichat-weblens'  # 固定使用AtriNexus模型
NETWORK_SEARCH_API_KEY = config.network_search.api_key
NETWORK_SEARCH_BASE_URL = config.network_search.base_url
