"""Redis队列管理器

使用Redis管理任务队列状态，替代MongoDB查询，提升性能。
"""
import logging
from typing import Optional, List
import redis
from redis.exceptions import RedisError

from config.settings import get_settings

logger = logging.getLogger(__name__)


class RedisQueueManager:
    """基于Redis的任务队列管理器

    使用Redis管理任务队列状态，提供原子性操作保证。
    """

    def __init__(self):
        self._settings = get_settings()
        self._redis = None

    @property
    def redis(self) -> redis.Redis:
        """获取Redis连接（延迟初始化）"""
        if self._redis is None:
            self._redis = redis.Redis.from_url(
                self._settings.celery_broker_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
        return self._redis

    @property
    def key_running(self) -> str:
        """运行中任务计数器key"""
        return "mindmap:queue:running"

    @property
    def key_waiting(self) -> str:
        """等待队列key"""
        return "mindmap:queue:waiting"

    @property
    def key_waiting_count(self) -> str:
        """等待队列计数器key"""
        return "mindmap:queue:waiting:count"

    def can_run_now(self) -> bool:
        """检查是否可以立即运行新任务

        Returns:
            True 如果可以立即运行，False 否则
        """
        try:
            running = self.redis.get(self.key_running) or "0"
            running_count = int(running)
            return running_count < self._settings.max_running_tasks
        except RedisError as e:
            logger.error(f"Redis查询失败: {e}")
            return False

    def add_to_waiting_queue(self, task_id: str) -> bool:
        """添加任务到等待队列

        Args:
            task_id: 任务ID

        Returns:
            True 如果成功添加，False 如果队列已满
        """
        try:
            waiting_count = self.redis.llen(self.key_waiting)
            if waiting_count >= self._settings.max_waiting_tasks:
                return False

            self.redis.rpush(self.key_waiting, task_id)
            self.redis.incr(self.key_waiting_count)
            logger.info(f"任务加入等待队列: {task_id}, 当前等待数: {waiting_count + 1}")
            return True
        except RedisError as e:
            logger.error(f"添加到等待队列失败: {e}")
            return False

    def schedule_next(self) -> Optional[str]:
        """从等待队列调度下一个任务

        Returns:
            任务ID，如果没有任务则返回None
        """
        try:
            task_id = self.redis.lpop(self.key_waiting)
            if task_id:
                self.redis.decr(self.key_waiting_count)
                logger.info(f"从等待队列调度任务: {task_id}")
            return task_id
        except RedisError as e:
            logger.error(f"调度任务失败: {e}")
            return None

    def increment_running(self) -> int:
        """增加运行中任务计数

        Returns:
            当前的运行中任务数
        """
        try:
            count = self.redis.incr(self.key_running)
            logger.info(f"运行中任务数增加: {count}")
            return count
        except RedisError as e:
            logger.error(f"增加运行中计数失败: {e}")
            return 0

    def decrement_running(self) -> int:
        """减少运行中任务计数

        Returns:
            当前的运行中任务数
        """
        try:
            count = self.redis.decr(self.key_running)
            if count < 0:
                self.redis.set(self.key_running, 0)
                count = 0
            logger.info(f"运行中任务数减少: {count}")
            return count
        except RedisError as e:
            logger.error(f"减少运行中计数失败: {e}")
            return 0

    def get_queue_status(self) -> dict:
        """获取队列状态

        Returns:
            {"running": int, "waiting": int, "max_running": int, "max_waiting": int}
        """
        try:
            running = int(self.redis.get(self.key_running) or "0")
            waiting = self.redis.llen(self.key_waiting)
            return {
                "running": running,
                "waiting": waiting,
                "max_running": self._settings.max_running_tasks,
                "max_waiting": self._settings.max_waiting_tasks
            }
        except RedisError as e:
            logger.error(f"获取队列状态失败: {e}")
            return {
                "running": 0,
                "waiting": 0,
                "max_running": self._settings.max_running_tasks,
                "max_waiting": self._settings.max_waiting_tasks
            }

    def get_waiting_queue(self, start: int = 0, end: int = -1) -> List[str]:
        """获取等待队列中的任务ID列表

        Args:
            start: 起始位置
            end: 结束位置，-1表示到末尾

        Returns:
            任务ID列表
        """
        try:
            return self.redis.lrange(self.key_waiting, start, end)
        except RedisError as e:
            logger.error(f"获取等待队列失败: {e}")
            return []

    def clear_waiting_queue(self) -> int:
        """清空等待队列

        Returns:
            清除的任务数量
        """
        try:
            count = self.redis.delete(self.key_waiting)
            self.redis.set(self.key_waiting_count, 0)
            logger.info(f"清空等待队列: {count} 个任务")
            return count
        except RedisError as e:
            logger.error(f"清空等待队列失败: {e}")
            return 0

    def reset_running_count(self, count: int = 0) -> bool:
        """重置运行中任务计数

        Args:
            count: 新的计数值

        Returns:
            是否成功
        """
        try:
            self.redis.set(self.key_running, count)
            logger.info(f"重置运行中计数: {count}")
            return True
        except RedisError as e:
            logger.error(f"重置运行中计数失败: {e}")
            return False

    def reset_all_queue_state(self) -> bool:
        """重置所有队列状态到初始状态
        
        Returns:
            是否成功
        """
        try:
            # 重置所有队列相关的键
            self.redis.set(self.key_running, 0)
            self.redis.delete(self.key_waiting)
            self.redis.set(self.key_waiting_count, 0)
            logger.info("已重置所有队列状态")
            return True
        except RedisError as e:
            logger.error(f"重置所有队列状态失败: {e}")
            return False

    def is_valid_task_id(self, task_id: str) -> bool:
        """验证任务ID是否有效（非空字符串）

        Args:
            task_id: 任务ID

        Returns:
            是否有效
        """
        return task_id is not None and isinstance(task_id, str) and len(task_id) > 0

    def init_from_mongo(self, running_count: int, waiting_task_ids: list) -> None:
        """从MongoDB初始化Redis队列状态

        在系统启动时调用，确保Redis状态与MongoDB一致。

        Args:
            running_count: MongoDB中运行中的任务数
            waiting_task_ids: MongoDB中等待中的任务ID列表
        """
        try:
            # 重置运行中计数
            self.redis.set(self.key_running, running_count)
            logger.info(f"初始化运行中计数: {running_count}")

            # 清空并重建等待队列
            self.redis.delete(self.key_waiting)
            self.redis.delete(self.key_waiting_count)
            if waiting_task_ids:
                # 只保留有效的任务ID
                valid_task_ids = [tid for tid in waiting_task_ids if self.is_valid_task_id(tid)]
                if valid_task_ids:
                    self.redis.rpush(self.key_waiting, *valid_task_ids)
                    logger.info(f"初始化等待队列: {len(valid_task_ids)} 个任务")
                else:
                    logger.info("等待队列为空")
            else:
                logger.info("等待队列为空")

            self.redis.set(self.key_waiting_count, len(waiting_task_ids) if waiting_task_ids else 0)
            
            # 验证初始化结果
            actual_running = int(self.redis.get(self.key_running) or "0")
            actual_waiting = self.redis.llen(self.key_waiting)
            logger.info(f"验证初始化结果 - 运行中: {actual_running}, 等待中: {actual_waiting}")

        except RedisError as e:
            logger.error(f"初始化Redis队列失败: {e}")


_global_redis_queue_manager: Optional[RedisQueueManager] = None


def get_redis_queue_manager() -> RedisQueueManager:
    """获取Redis队列管理器单例

    Returns:
        RedisQueueManager: 队列管理器实例
    """
    global _global_redis_queue_manager
    if _global_redis_queue_manager is None:
        _global_redis_queue_manager = RedisQueueManager()
    return _global_redis_queue_manager
