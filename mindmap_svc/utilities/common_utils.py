import base64
import binascii
import json
import random
import sys
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Union

import pytz
import requests
import os

# 缓存配置文件内容
_config_cache: Dict[str, Any] = {}


def get_config(key_path: str, default: Any = None) -> Any:
    """
    从appconfig.json中获取配置值
    
    Args:
        key_path: 配置路径，支持点分隔，如 "env.KB_MINIO_ENDPOINT"
        default: 默认值
        
    Returns:
        配置值或默认值
    """
    global _config_cache
    
    # 加载配置文件（仅首次）
    if not _config_cache:
        try:
            # 获取项目根目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            config_path = os.path.join(project_root, "appconfig.json")
            
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    _config_cache = json.load(f)
        except Exception as e:
            print(f"Loading config file failed: {e}")
            _config_cache = {}
    
    # 按路径获取配置值
    keys = key_path.split('.')
    value = _config_cache
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    # 优先从环境变量获取（如果是env配置）
    if keys[0] == "env" and len(keys) > 1:
        env_key = keys[-1]
        env_value = os.environ.get(env_key)
        if env_value is not None:
            return env_value
    
    return value if value is not None else default


def decode_token(token) -> Any | None:
    if token is None:
        return None
    _k_sys = os.environ['KB_API_SERVICE_AUTHENTICATION']
    token_str = token[7:]
    # account_service="http://{}/users/userinfo?token={}".format(_account_mgr,token)
    sys_service = "http://{}/token/userinfo?token={}".format(_k_sys, token_str)
    headers = {
        "Authorization": token,
        "access-key": "1effa9c3b68447c7b8dcac84ccaebd32"
    }
    try:
        rp = requests.get(url=sys_service, headers=headers)
        if rp.ok:
            rp_json = rp.json()
            if rp_json['code'] == 200:
                return rp_json['data']
            else:
                print("Parsing token exception occurred: {}".format(json.dumps(rp_json, ensure_ascii=False)))
        else:
            print("Invoking k-sys exception occurred:{}".format(rp.status_code))
    except Exception as e:
        print("Parsing token exception occurred: {}".format(e))

    return None


def string_to_dict(input_string: str) -> Dict[str, Any]:
    """
    将输入的字符串转换为字典对象。

    如果字符串为空、不合法或转换失败，返回空字典。
    不会抛出异常。

    Args:
        input_string (str): 要转换的输入字符串

    Returns:
        Dict[str, Any]: 转换后的字典，如果转换失败则返回空字典
    """
    # 检查输入是否为空
    if not input_string or not isinstance(input_string, str):
        return {}

    # 尝试转换字符串为字典
    try:
        # 去除字符串前后的空白字符
        input_string = input_string.strip()

        # 尝试解析 JSON 字符串
        result = json.loads(input_string)

        # 确保结果是字典类型
        if not isinstance(result, dict):
            return {}

        return result

    except Exception as e:
        # 捕获所有可能的异常
        print("Converting string to dict exception occurred: {}".format(e))
    return {}


def generate_guid() -> str:
    """
    生成一个 GUID 字符串。

    该函数生成一个标准的 GUID，然后移除所有的破折号 ('-')，
    并将所有字符转换为小写。

    Returns:
        str: 自定义格式的 GUID 字符串
    """
    # 生成标准的 GUID
    original_guid = uuid.uuid4()

    # 将 GUID 转换为字符串，移除破折号，并转换为小写
    custom_guid = str(original_guid).replace('-', '').lower()

    return custom_guid


def get_formatted_current_time() -> str:
    """
    返回当前时间的格式化字符串。

    返回的时间格式为：YYYY-MM-DD HH:MM:SS
    例如：2024-09-24 12:00:00

    Returns:
        str: 格式化的当前时间字符串
    """
    # 获取当前时间
    current_time = datetime.now()

    # 将当前时间格式化为指定的字符串格式
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

    return formatted_time


def get_dict_value(data: Dict[str, Any], key: str) -> Optional[Any]:
    """
    从给定的字典中获取指定键的值。

    参数:
    data (Dict[str, Any]): 输入的字典
    key (str): 要查找的键

    返回:
    Optional[Any]: 如果指定键存在且有效，返回其值；否则返回None

    异常:
    TypeError: 如果输入的data不是字典类型，或者key不是字符串类型
    """
    # 检查输入是否为字典类型
    if not isinstance(data, dict):
        raise TypeError("The 'data' parameter must be a dictionary")

    # 检查 key 是否为字符串类型
    if not isinstance(key, str):
        raise TypeError("The 'key' parameter must be a string")

    # 检查字典是否为空（长度为0）
    if len(data) == 0:
        return None

    # 检查指定的键是否在字典中
    if key not in data:
        return None

    # 获取指定键的值
    value = data[key]

    # 检查值是否为None或空字符串
    if value is None or (isinstance(value, str) and not value.strip()):
        return None

    # 返回有效的值
    return value


def string_to_base64(input_string: str) -> str:
    """
    将输入的字符串转换为Base64编码。

    参数:
    input_string (str): 需要转换的输入字符串

    返回:
    str: Base64编码后的字符串

    异常:
    TypeError: 如果输入不是字符串类型
    """
    # 数据验证
    if not isinstance(input_string, str):
        raise TypeError("输入必须是字符串类型")

    try:
        # 将字符串编码为bytes，然后转换为Base64
        bytes_data = input_string.encode('utf-8')
        base64_bytes = base64.b64encode(bytes_data)
        return base64_bytes.decode('ascii')
    except Exception as e:
        # 捕获并重新抛出异常，提供更多上下文信息
        raise Exception(f"转换字符串到Base64时发生错误: {str(e)}")


def base64_to_string(base64_string: str) -> str:
    """
    将Base64编码的字符串解码为普通字符串。

    参数:
    base64_string (str): Base64编码的字符串

    返回:
    str: 解码后的普通字符串

    异常:
    TypeError: 如果输入不是字符串类型
    ValueError: 如果输入不是有效的Base64编码
    """
    # 数据验证
    if not isinstance(base64_string, str):
        raise TypeError("输入必须是字符串类型")

    try:
        # 解码Base64字符串
        bytes_data = base64.b64decode(base64_string, validate=True)
        return bytes_data.decode('utf-8')
    except binascii.Error:
        raise ValueError("输入不是有效的Base64编码")
    except UnicodeDecodeError:
        raise ValueError("Base64解码后的数据不是有效的UTF-8编码")
    except Exception as e:
        # 捕获并重新抛出异常，提供更多上下文信息
        raise Exception(f"从Base64转换到字符串时发生错误: {str(e)}")


def string_starts_with(input_string, prefix):
    """
    检查input_string是否以prefix开头。

    参数:
    input_string (str): 要检查的输入字符串
    prefix (str): 要匹配的前缀字符串

    返回:
    bool: 如果input_string以prefix开头，返回True；否则返回False

    异常:
    TypeError: 如果输入的参数不是字符串类型
    ValueError: 如果输入的字符串为空
    """
    # 参数类型检查
    if not isinstance(input_string, str) or not isinstance(prefix, str):
        raise TypeError("输入必须是字符串类型")

    # 空字符串检查
    if not input_string or not prefix:
        raise ValueError("输入不能为空字符串")

    # 使用内置的startswith方法进行检查
    return input_string.startswith(prefix)


def get_random_int(start: Union[int, str], end: Union[int, str]) -> int:
    """
    从闭区间[start, end]中随机选择一个整数

    参数:
        start: 区间起始值，可以是整数或者字符串
        end: 区间结束值，可以是整数或者字符串

    返回:
        int: 区间内的随机整数，如果输入无效则返回start转换后的值或默认值1
        当start和end相等时，直接返回start值
    """
    # 转换参数为整数，处理可能的类型转换错误
    try:
        start_int = int(float(str(start)))
        end_int = int(float(str(end)))
    except (ValueError, TypeError):
        return 1  # 如果转换失败返回默认值1

    # 如果start和end相等，直接返回start值
    if start_int == end_int:
        return start_int

    # 确保区间有效（start <= end）
    if start_int > end_int:
        start_int, end_int = end_int, start_int  # 交换确保start <= end

    # 处理超出Python整数范围的情况
    start_int = max(start_int, -sys.maxsize - 1)
    end_int = min(end_int, sys.maxsize)

    # 使用random.randint生成随机数
    try:
        return random.randint(start_int, end_int)
    except ValueError:  # 以防万一还有其他异常情况
        return start_int


def encode_image(image_path: str) -> str:
    """Base64编码本地图片文件"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def convert_datetime_of_dict(obj):
    """转换字典中的datetime为字符串格式"""
    if isinstance(obj, dict):
        return {key: convert_datetime_of_dict(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_datetime_of_dict(item) for item in obj]
    elif isinstance(obj, datetime):
        # 确保datetime有时区信息并格式化
        if obj.tzinfo is None:
            obj = obj.replace(tzinfo=pytz.UTC)
        return obj.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    return obj