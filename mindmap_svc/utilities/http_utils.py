from common.enums import APIMethodEnum
from typing import Dict, List, Any, Optional
from collections import Counter
import asyncio
import httpx
import logging
import pandas as pd
from io import StringIO

logger = logging.getLogger(__name__)


class APIConfig:
    """API配置类"""

    # API服务器配置
    BASE_URL = "http://your-api-server"
    TIMEOUT = 60
    MAX_RETRY = 3

    # API端点
    ENDPOINT_TEST_POINT_DATA = "/api/v1/timeseries/data"
    ENDPOINT_DIAGNOSIS_ANALYZE = "/api/v1/diagnosis/analyze"


class HTTPClient:
    """通用HTTP客户端工具类"""

    def __init__(self, base_url: str = "", timeout: int = 60, max_retry: int = 3):
        """
        初始化HTTP客户端

        Args:
            base_url: API基础URL
            timeout: 超时时间（秒）
            max_retry: 最大重试次数
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retry = max_retry

    async def request(
            self,
            method: APIMethodEnum,
            url: str,
            params: Optional[Dict] = None,
            json_data: Optional[Dict] = None,
            headers: Optional[Dict] = None,
            retry: bool = True
    ) -> Dict[str, Any]:
        """
        通用HTTP请求方法

        Args:
            method: HTTP方法
            url: 请求URL（可以是完整URL或相对路径）
            params: URL参数
            json_data: JSON请求体
            headers: 请求头
            retry: 是否启用重试

        Returns:
            Dict: 响应数据

        Raises:
            Exception: 请求失败
        """
        # 构建完整URL
        if not url.startswith('http'):
            url = f"{self.base_url}{url}" if url.startswith('/') else f"{self.base_url}/{url}"

        # 默认请求头
        default_headers = {"Content-Type": "application/json"}
        if headers:
            default_headers.update(headers)

        retry_count = 0
        last_error = None

        while retry_count < (self.max_retry if retry else 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    logger.debug(f"HTTP {method.value} {url} (第 {retry_count + 1} 次)")

                    # 发送请求
                    response = await client.request(
                        method=method.value,
                        url=url,
                        params=params,
                        json=json_data,
                        headers=default_headers
                    )

                    # 检查状态码
                    response.raise_for_status()

                    # 解析响应
                    result = response.json()

                    # 统一响应格式处理
                    return self._parse_response(result)

            except httpx.TimeoutException as e:
                last_error = f"请求超时: {str(e)}"
                logger.warning(f"{last_error} (第 {retry_count + 1}/{self.max_retry} 次)")

            except httpx.HTTPStatusError as e:
                last_error = f"HTTP错误 {e.response.status_code}: {str(e)}"
                logger.warning(f"{last_error} (第 {retry_count + 1}/{self.max_retry} 次)")

            except Exception as e:
                last_error = f"请求异常: {str(e)}"
                logger.warning(f"{last_error} (第 {retry_count + 1}/{self.max_retry} 次)")

            retry_count += 1

            # 如果还需要重试，则等待
            if retry and retry_count < self.max_retry:
                await asyncio.sleep(2)

        # 重试失败
        error_msg = f"HTTP请求失败 ({url}): {last_error}"
        logger.error(error_msg)
        raise Exception(error_msg)

    def _parse_response(self, result: Dict) -> Dict[str, Any]:
        """
        解析统一响应格式

        Args:
            result: 原始响应数据

        Returns:
            Dict: 解析后的数据

        Raises:
            Exception: 响应错误
        """
        # 处理标准响应格式: {code: 200, message: "success", data: {...}}
        if "code" in result:
            if result["code"] != 200:
                error_msg = result.get("message", "未知错误")
                raise Exception(f"API返回错误: {error_msg}")
            return result.get("data", {})

        # 直接返回原始数据
        return result

    async def get(self, url: str, params: Optional[Dict] = None, **kwargs) -> Dict:
        """GET请求"""
        return await self.request(APIMethodEnum.GET, url, params=params, **kwargs)

    async def post(self, url: str, json_data: Optional[Dict] = None, **kwargs) -> Dict:
        """POST请求"""
        return await self.request(APIMethodEnum.POST, url, json_data=json_data, **kwargs)

    async def put(self, url: str, json_data: Optional[Dict] = None, **kwargs) -> Dict:
        """PUT请求"""
        return await self.request(APIMethodEnum.PUT, url, json_data=json_data, **kwargs)

    async def delete(self, url: str, **kwargs) -> Dict:
        """DELETE请求"""
        return await self.request(APIMethodEnum.DELETE, url, **kwargs)


class TestPointConfig:
    """测点配置枚举类"""

    # 测点配置字典：{列名: (测点类型, 单位, 正常范围下限, 正常范围上限, 正常状态描述)}
    CONFIG = {
        # ========== 原有配置（14个测点）==========
        '导叶开度(%)': {
            'type': '导叶控制',
            'unit': '%',
            'normal_min': 70.0,
            'normal_max': 73.0,
            'normal_state': '开度稳定在70-73%，偏差<±0.5%'
        },
        '主接力器位移(mm)': {
            'type': '导叶控制',
            'unit': 'mm',
            'normal_min': 140.0,
            'normal_max': 145.0,
            'normal_state': '位移140-145mm，线性响应无卡滞'
        },
        '接力器行程时间(s)': {
            'type': '导叶控制',
            'unit': 's',
            'normal_min': 8.0,
            'normal_max': 12.0,
            'normal_state': '全行程8-12秒，动作均匀'
        },
        '主接力器油压(MPa)': {
            'type': '油压系统',
            'unit': 'MPa',
            'normal_min': 5.0,
            'normal_max': 5.3,
            'normal_state': '压力5.0-5.3MPa，波动<±0.1MPa'
        },
        '调速器油压(MPa)': {
            'type': '油压系统',
            'unit': 'MPa',
            'normal_min': 6.2,
            'normal_max': 6.4,
            'normal_state': '恒定6.3MPa，偏差<±0.1MPa'
        },
        '压油罐压力(MPa)': {
            'type': '油压系统',
            'unit': 'MPa',
            'normal_min': 6.2,
            'normal_max': 6.4,
            'normal_state': '稳定6.3MPa，补压及时'
        },
        '回油箱压力(MPa)': {
            'type': '油压系统',
            'unit': 'MPa',
            'normal_min': 0.1,
            'normal_max': 0.2,
            'normal_state': '压力0.1-0.2MPa，回油顺畅'
        },
        '事故油压(MPa)': {
            'type': '油压系统',
            'unit': 'MPa',
            'normal_min': 6.2,
            'normal_max': 6.4,
            'normal_state': '压力6.3MPa，建立迅速'
        },
        '油泵出口压力(MPa)': {
            'type': '油压系统',
            'unit': 'MPa',
            'normal_min': 6.4,
            'normal_max': 6.6,
            'normal_state': '压力6.5MPa，稳定输出'
        },
        '滤油器差压(kPa)': {
            'type': '油路监测',
            'unit': 'kPa',
            'normal_min': 20.0,
            'normal_max': 50.0,
            'normal_state': '差压20-50kPa，滤网清洁'
        },
        '压油罐液位(m)': {
            'type': '液位监测',
            'unit': 'm',
            'normal_min': 2.5,
            'normal_max': 3.0,
            'normal_state': '液位2.5-3.0m，稳定无波动'
        },
        '回油箱液位(m)': {
            'type': '液位监测',
            'unit': 'm',
            'normal_min': 1.5,
            'normal_max': 2.0,
            'normal_state': '液位1.5-2.0m，回油正常'
        },
        '调速器油温(℃)': {
            'type': '温度监测',
            'unit': '℃',
            'normal_min': 35.0,
            'normal_max': 45.0,
            'normal_state': '温度35-45℃，散热良好'
        },
        '压油罐油温(℃)': {
            'type': '温度监测',
            'unit': '℃',
            'normal_min': 35.0,
            'normal_max': 40.0,
            'normal_state': '温度35-40℃，无异常升温'
        },

        # ========== 新增配置（15个测点）==========
        '操作机构电流(A)': {
            'type': '电气监测',
            'unit': 'A',
            'normal_min': 15.0,
            'normal_max': 25.0,
            'normal_state': '电流15-25A，负载正常'
        },
        '各导叶开度偏差(%)': {
            'type': '导叶控制',
            'unit': '%',
            'normal_min': 0.0,
            'normal_max': 1.5,
            'normal_state': '各导叶偏差<±1.5%，同步性良好'
        },
        '导叶间隙(mm)': {
            'type': '机械磨损',
            'unit': 'mm',
            'normal_min': 0.5,
            'normal_max': 1.0,
            'normal_state': '间隙0.5-1.0mm，磨损正常'
        },
        '机组效率(%)': {
            'type': '性能指标',
            'unit': '%',
            'normal_min': 92.0,
            'normal_max': 94.0,
            'normal_state': '效率92-94%，运行高效'
        },
        '导叶区域噪声(dB)': {
            'type': '振动噪声',
            'unit': 'dB',
            'normal_min': 65.0,
            'normal_max': 75.0,
            'normal_state': '噪声65-75dB，无异常声响'
        },
        '水导轴承温度(℃)': {
            'type': '轴承监测',
            'unit': '℃',
            'normal_min': 45.0,
            'normal_max': 55.0,
            'normal_state': '温度45-55℃，润滑良好'
        },
        '推力轴承温度(℃)': {
            'type': '轴承监测',
            'unit': '℃',
            'normal_min': 50.0,
            'normal_max': 60.0,
            'normal_state': '温度50-60℃，承载正常'
        },
        '轴向位移(mm)': {
            'type': '位移监测',
            'unit': 'mm',
            'normal_min': 0.3,
            'normal_max': 0.5,
            'normal_state': '位移±0.3-0.5mm，推力正常'
        },
        '径向位移(mm)': {
            'type': '位移监测',
            'unit': 'mm',
            'normal_min': 0.05,
            'normal_max': 0.15,
            'normal_state': '位移≤0.15mm，对中良好'
        },
        '润滑油压力(MPa)': {
            'type': '润滑系统',
            'unit': 'MPa',
            'normal_min': 0.15,
            'normal_max': 0.20,
            'normal_state': '压力0.15-0.20MPa，供油充足'
        },
        '润滑油流量(L/min)': {
            'type': '润滑系统',
            'unit': 'L/min',
            'normal_min': 180.0,
            'normal_max': 220.0,
            'normal_state': '流量180-220L/min，循环正常'
        },
        '润滑油温度(℃)': {
            'type': '润滑系统',
            'unit': '℃',
            'normal_min': 45.0,
            'normal_max': 50.0,
            'normal_state': '温度45-50℃，冷却正常'
        },
        '金属颗粒浓度(ppm)': {
            'type': '油液监测',
            'unit': 'ppm',
            'normal_min': 1.0,
            'normal_max': 5.0,
            'normal_state': '浓度<5ppm，磨损轻微'
        },
        '漏油检测信号': {
            'type': '泄漏监测',
            'unit': '',
            'normal_min': 0,
            'normal_max': 0,
            'normal_state': '无漏油信号，密封完好'
        }
    }

    @classmethod
    def get_config(cls, column_name: str) -> Dict:
        """获取测点配置"""
        return cls.CONFIG.get(column_name, {
            'type': '未分类',
            'unit': '',
            'normal_min': None,
            'normal_max': None,
            'normal_state': '未配置正常范围'
        })

    @classmethod
    def add_testpoint(cls, column_name: str, test_point_type: str, unit: str,
                      normal_min: float, normal_max: float, normal_state: str):
        """动态添加新测点配置"""
        cls.CONFIG[column_name] = {
            'type': test_point_type,
            'unit': unit,
            'normal_min': normal_min,
            'normal_max': normal_max,
            'normal_state': normal_state
        }


class DiagnosisAPIService:
    """诊断系统API服务类"""

    def __init__(self, http_client: HTTPClient):
        """
        初始化诊断API服务

        Args:
            http_client: HTTP客户端实例
        """
        self.http_client = http_client

    async def get_test_point_data(
            self,
            unit_code: str,
            component_code: str
    ) -> pd.DataFrame:
        """
        获取测点时序数据

        Args:
            unit_code: 机组编号
            component_code: 部件编号

        Returns:
            pd.DataFrame: 测点时序数据

        Raises:
            Exception: 获取数据失败
        """
        try:
            logger.info(f"获取测点数据: 机组={unit_code}, 部件={component_code}")

            # 调用API
            result = await self.http_client.post(
                url="/api/v1/timeseries/data",
                json_data={
                    "unit_code": unit_code,
                    "component_code": component_code
                }
            )

            # 解析为DataFrame
            df = self._parse_to_dataframe(result)

            # 数据验证
            if df.empty:
                raise Exception("获取到的数据为空")

            if "timestamp" not in df.columns:
                raise Exception("数据缺少timestamp列")

            # 转换时间戳
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            logger.info(f"成功获取数据: {len(df)}条记录, {len(df.columns) - 1}个测点")
            return df

        except Exception as e:
            logger.error(f"获取测点数据失败: {str(e)}")
            raise

    async def analyze_data(self, df: pd.DataFrame, time_column: str = '时间戳', sensitivity: float = 1.0) \
            -> List[Dict[str, Any]]:
        """
        通用测点数据异常分析函数

        参数:
            df: CSV_DATA
            time_column: 时间列名称，默认'时间戳'
            sensitivity: 异常检测灵敏度，1.0为标准，值越小越敏感

        返回:
            包含所有测点异常时间段分析结果的列表
        """
        # 验证时间列是否存在
        if time_column not in df.columns:
            raise ValueError(f"时间列 '{time_column}' 不存在于CSV文件中")

        # 3. 转换时间列为datetime类型
        df[time_column] = pd.to_datetime(df[time_column])

        # 4. 初始化结果列表
        results = []

        # 5. 遍历每个测点列（排除时间列）
        test_point_columns = [col for col in df.columns if col != time_column]

        for column in test_point_columns:
            # 获取测点配置
            config = TestPointConfig.get_config(column)
            test_point_type = config['type']
            unit = config['unit']
            normal_min = config['normal_min']
            normal_max = config['normal_max']
            normal_state = config['normal_state']

            # 跳过配置缺失或数据全为空的列
            if normal_min is None or normal_max is None:
                continue

            # 获取该测点的数据
            data_series = df[column].dropna()
            if len(data_series) == 0:
                continue

            # 6. 检测异常时间段
            exception_slots = self.detect_exception_periods(
                df=df,
                column=column,
                time_column=time_column,
                normal_min=normal_min,
                normal_max=normal_max,
                unit=unit,
                sensitivity=sensitivity
            )

            # 7. 构建该测点的结果（只输出有异常的测点）
            test_point_result = {
                'test_point_id': column,
                'test_point_type': test_point_type,
                'normal_state': normal_state,
                'exception_time_slot': exception_slots if exception_slots else []
            }

            results.append(test_point_result)

        return results

    def detect_exception_periods(self, df: pd.DataFrame,
                                 column: str,
                                 time_column: str,
                                 normal_min: float,
                                 normal_max: float,
                                 unit: str,
                                 sensitivity: float = 1.0,
                                 min_duration_points: int = 3) -> List[Dict[str, Any]]:
        """
        检测异常时间段

        参数:
            df: 数据框
            column: 要分析的列名
            time_column: 时间列名
            normal_min: 正常范围下限
            normal_max: 正常范围上限
            unit: 单位
            sensitivity: 灵敏度系数
            min_duration_points: 最小异常持续点数（避免单点噪声）

        返回:
            异常时间段列表
        """

        # 创建异常标记
        # 使用动态阈值：考虑正常范围的一定比例扩展
        range_width = normal_max - normal_min
        tolerance = range_width * 0.001 * sensitivity  # 10%容差

        lower_threshold = normal_min - tolerance
        upper_threshold = normal_max + tolerance

        # 标记异常点
        df_copy = df[[time_column, column]].copy()
        df_copy['is_exception'] = (
                (df_copy[column] < lower_threshold) |
                (df_copy[column] > upper_threshold)
        )

        # 查找连续异常段
        exception_periods = []
        in_exception = False
        start_idx = None

        for idx, row in df_copy.iterrows():
            if row['is_exception'] and not in_exception:
                # 异常开始
                in_exception = True
                start_idx = idx
            elif not row['is_exception'] and in_exception:
                # 异常结束
                end_idx = idx - 1

                # 检查持续时间是否满足最小点数要求
                if end_idx - start_idx + 1 >= min_duration_points:
                    period_data = df_copy.loc[start_idx:end_idx, column]

                    # 计算该时间段的统计指标
                    stats = self.calculate_statistics(period_data)

                    exception_periods.append({
                        'start_time': df_copy.loc[start_idx, time_column].strftime('%Y-%m-%d %H:%M:%S'),
                        'end_time': df_copy.loc[end_idx, time_column].strftime('%Y-%m-%d %H:%M:%S'),
                        'highest': f"{stats['max']:.2f}{unit}",
                        'minimum': f"{stats['min']:.2f}{unit}",
                        'average': f"{stats['mean']:.2f}{unit}",
                        'range': f"{stats['range']:.2f}{unit}",
                        'coefficient_of_variation': f"{stats['cv']:.2f}%"
                    })

                in_exception = False
                start_idx = None

        # 处理最后一段（如果数据以异常结束）
        if in_exception and start_idx is not None:
            end_idx = df_copy.index[-1]
            if end_idx - start_idx + 1 >= min_duration_points:
                period_data = df_copy.loc[start_idx:end_idx, column]
                stats = self.calculate_statistics(period_data)

                exception_periods.append({
                    'start_time': df_copy.loc[start_idx, time_column].strftime('%Y-%m-%d %H:%M:%S'),
                    'end_time': df_copy.loc[end_idx, time_column].strftime('%Y-%m-%d %H:%M:%S'),
                    'highest': f"{stats['max']:.2f}{unit}",
                    'minimum': f"{stats['min']:.2f}{unit}",
                    'average': f"{stats['mean']:.2f}{unit}",
                    'range': f"{stats['range']:.2f}{unit}",
                    'coefficient_of_variation': f"{stats['cv']:.2f}%"
                })

        return exception_periods

    def calculate_statistics(self, data_series: pd.Series) -> Dict[str, float]:
        """
        计算统计指标

        参数:
            data_series: 数据序列
            unit: 单位

        返回:
            包含各项统计指标的字典
        """
        mean = data_series.mean()
        std = data_series.std()
        min_val = data_series.min()
        max_val = data_series.max()
        range_val = max_val - min_val

        # 变异系数 = (标准差 / 均值) * 100%
        cv = (std / mean * 100) if mean != 0 else 0

        return {
            'mean': mean,
            'std': std,
            'min': min_val,
            'max': max_val,
            'range': range_val,
            'cv': cv
        }

    def _parse_to_dataframe(self, data: Any) -> pd.DataFrame:
        """
        将API返回数据解析为DataFrame

        Args:
            data: API返回的数据

        Returns:
            pd.DataFrame: 解析后的DataFrame
        """
        # CSV字符串格式
        if isinstance(data, str):
            return pd.read_csv(StringIO(data))

        # 字典列表格式
        if isinstance(data, list):
            return pd.DataFrame(data)

        # 嵌套字典格式
        if isinstance(data, dict):
            if "csv_data" in data:
                return pd.read_csv(StringIO(data["csv_data"]))
            elif "records" in data:
                return pd.DataFrame(data["records"])
            else:
                return pd.DataFrame(data)

        raise ValueError(f"不支持的数据格式: {type(data)}")

    async def initial_diagnosis(self, analysis_result: List[Dict[str, Any]]) -> list:
        """根据测点分析结果，逐步诊断故障类型"""
        failure_mode_list = []
        mapping = [
            {
                "failure_mode": "连杆变形",
                "mode_measurement_point_number": 5,
                "mode_measurement_point": [
                    "导叶开度(%)",
                    "主接力器位移(mm)",
                    "接力器行程时间(s)",
                    "主接力器油压(MPa)",
                    "调速器油压(MPa)"
                ]
            },
            {
                "failure_mode": "油路堵塞",
                "mode_measurement_point_number": 9,
                "mode_measurement_point": [
                    "压油罐压力(MPa)",
                    "回油箱压力(MPa)",
                    "事故油压(MPa)",
                    "油泵出口压力(MPa)",
                    "滤油器差压(kPa)",
                    "压油罐液位(m)",
                    "回油箱液位(m)",
                    "调速器油温(℃)",
                    "压油罐油温(℃)"
                ]
            },
            {
                "failure_mode": "锈蚀卡死",
                "mode_measurement_point_number": 4,
                "mode_measurement_point": [
                    "导叶开度(%)",
                    "接力器行程时间(s)",
                    "主接力器油压(MPa)",
                    "操作机构电流(A)"
                ]
            },
            {
                "failure_mode": "导叶体磨损",
                "mode_measurement_point_number": 5,
                "mode_measurement_point": [
                    "导叶开度(%)",
                    "导叶间隙(mm)",
                    "机组效率(%)",
                    "导叶区域噪声(dB)",
                    "水导轴承温度(℃)"
                ]
            },
            {
                "failure_mode": "接力器泄漏",
                "mode_measurement_point_number": 7,
                "mode_measurement_point": [
                    "主接力器位移(mm)",
                    "主接力器油压(MPa)",
                    "压油罐压力(MPa)",
                    "压油罐液位(m)",
                    "回油箱液位(m)",
                    "接力器行程时间(s)",
                    "调速器油温(℃)"
                ]
            },
            {
                "failure_mode": "轴承销轴磨损",
                "mode_measurement_point_number": 7,
                "mode_measurement_point": [
                    "水导轴承温度(℃)",
                    "轴向位移(mm)",
                    "径向位移(mm)",
                    "润滑油压力(MPa)",
                    "润滑油流量(L/min)",
                    "润滑油温度(℃)",
                    "金属颗粒浓度(ppm)"
                ]
            }
        ]
        try:
            unique_list = []
            for i in analysis_result:
                if len(i["exception_time_slot"]) != 0:
                    for item in mapping:
                        if i["test_point_id"] in item["mode_measurement_point"]:
                            failure_mode_list.append(item["failure_mode"])
            if len(failure_mode_list) != 0:
                # 按需更改  保留列表中故障类型最多的元素
                count = Counter(failure_mode_list) # {'接力器泄漏': 4, '锈蚀卡死': 3, '连杆变形': 2, '油路堵塞': 2}
                list(count.keys()) # ['接力器泄漏', '锈蚀卡死', '连杆变形', '油路堵塞']
                result = {}
                for i in list(count.keys()):
                    for l in mapping:
                        if i == l["failure_mode"]:
                            result[i] = count[i] / l["mode_measurement_point_number"]
                # print(result)
                # # print(count["连杆变形"])
                # print(failure_mode_list)
                max_key = max(result, key=result.get) # 初步诊断故障
                unique_list.append(max_key)
            if len(failure_mode_list) == 0:
                unique_list.append("正常运行")

            # print(unique_list)
            return unique_list
        except Exception as e:
            raise e
