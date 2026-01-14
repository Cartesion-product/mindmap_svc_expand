"""API 服务器模块

提供 FastAPI 应用服务器的初始化和启动功能。
"""
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import mindmap_router # 新增引用
from middleware.error_handler import setup_exception_handlers
from common import constants
from utilities.log_manager import LogManager


class ApiServer:
    """API 服务器类

    负责 FastAPI 应用的初始化、中间件配置、路由注册和启动。
    """

    def __init__(self, logger: LogManager, settings=None):
        """初始化 API 服务器

        Args:
            logger: 日志管理器实例
            settings: 配置管理实例（可选）
        """
        self.logger = logger
        self.settings = settings

        self.app = FastAPI(
            title=constants.SERVICE_TITLE,
            docs_url='/api-doc',
            description=constants.SERVICE_DESCRIPTION,
            version=constants.SERVICE_VERSION
        )

        self._initialize()

    def _initialize(self):
        """初始化应用服务器"""

        # 配置 CORS 中间件
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # 设置错误处理器
        setup_exception_handlers(self.app)

        # 注册路由
        self.app.include_router(
            mindmap_router.router,
            # prefix=constants.API_PREFIX
        )

        # 根路径
        @self.app.get("/")
        async def root():
            return {
                "message": f"{constants.SERVICE_TITLE} API Server",
                "status": "running",
                "version": constants.SERVICE_VERSION
            }

        # 启动事件
        @self.app.on_event("startup")
        async def startup_event():
            self.logger.info("正在初始化数据库连接...")
            try:
                from db.mongo import get_mongo_client
                client = get_mongo_client()
                client.ensure_indexes()
                self.logger.info("数据库初始化完成")
            except Exception as e:
                self.logger.error(f"数据库初始化失败: {e}")
                # 不中断启动流程，允许服务继续启动但功能受限
                pass

            self.logger.info("正在初始化Redis队列...")
            try:
                from common.redis_manager import get_redis_queue_manager
                from repositories.user_paper_repo import get_user_paper_repo
                from repositories.system_paper_repo import get_system_paper_repo
                from common.enums import TaskStatusEnum
                import redis

                queue_manager = get_redis_queue_manager()

                # 只有在数据库连接成功时才进行后续初始化
                if 'client' in locals() and client:
                    user_repo = get_user_paper_repo()
                    system_repo = get_system_paper_repo()

                    # 清理Celery队列中的所有消息，避免重启后重复执行
                    try:
                        redis_client = redis.Redis.from_url(
                            queue_manager._settings.celery_broker_url,
                            decode_responses=False
                        )
                        # 检查Redis连接
                        redis_client.ping()

                        # 清理所有相关的队列
                        celery_queues = ["celery", "mindmap"]
                        for queue in celery_queues:
                            # 获取队列长度
                            queue_length = redis_client.llen(queue)
                            if queue_length > 0:
                                self.logger.info(f"发现 {queue} 队列中有 {queue_length} 个待处理任务，正在清理...")

                                # 逐个移除队列中的任务
                                for _ in range(queue_length):
                                    redis_client.brpop(queue, timeout=1)

                                self.logger.info(f"已清理 {queue} 队列中的 {queue_length} 个任务")
                    except Exception as e:
                        self.logger.warning(f"清理Celery队列失败: {e}")

                    # 在初始化前重置所有队列状态，确保干净的初始状态
                    queue_manager.reset_all_queue_state()

                    # 服务启动时，删除所有无结果的系统记录
                    # 查找并删除所有file_path为空的系统记录
                    empty_results = system_repo.find_empty_results()
                    if empty_results:
                        deleted_count = system_repo.delete_empty_results()
                        self.logger.info(
                            f"已删除 {deleted_count} 个无结果的系统记录"
                        )
                        for record in empty_results:
                            self.logger.info(
                                f"  - paper_id={record.paper_id}, "
                                f"agent_type={record.agent_type}, source={record.source}"
                            )

                    # 服务重启时，将所有"运行中"的任务标记为失败（因为它们已被中断）
                    running_tasks = user_repo.find_many(
                        {"status": TaskStatusEnum.RUNNING.value},
                        limit=1000
                    )
                    for task in running_tasks:
                        user_repo.mark_failed(task.result_id, "任务因服务重启而中断")
                        self.logger.info(f"已将中断的任务标记为失败: {task.result_id}")

                    # 服务刚启动时，运行中计数应为0
                    running_count = 0

                    # 获取等待中的任务
                    waiting_tasks = user_repo.find_many(
                        {"status": TaskStatusEnum.WAITING.value},
                        limit=1000,
                        sort=[("created_time", 1)]
                    )

                    # 根据配置决定如何处理等待队列
                    if self.settings.reset_waiting_on_restart:
                        # 将等待任务标记为失败
                        for task in waiting_tasks:
                            user_repo.mark_failed(task.result_id, "任务因服务重启而取消")
                            self.logger.info(f"已将等待任务标记为失败: {task.result_id}")
                        waiting_task_ids = []
                    else:
                        # 保留等待任务，稍后重新调度
                        waiting_task_ids = [task.result_id for task in waiting_tasks]

                    # 初始化Redis队列状态
                    queue_manager.init_from_mongo(running_count, waiting_task_ids)

                    self.logger.info(
                        f"Redis队列初始化完成 - 运行中: {running_count}, 等待中: {len(waiting_task_ids)}"
                    )

                    # 如果有等待任务，触发调度
                    if waiting_task_ids:
                        from services.task_service import get_task_service
                        task_service = get_task_service()
                        task_service.schedule_from_waiting_queue()
                        self.logger.info("已触发等待任务调度")
                else:
                    self.logger.warning("数据库连接失败，跳过Redis队列高级初始化")
            except Exception as e:
                self.logger.error(f"Redis队列初始化失败: {e}")
                # 不中断启动流程，允许服务继续启动但功能受限
                pass

        # 关闭事件
        @self.app.on_event("shutdown")
        async def shutdown_event():
            self.logger.info("正在关闭数据库连接...")
            try:
                from db.mongo import get_mongo_client
                get_mongo_client().disconnect()
                self.logger.info("数据库连接已关闭")
            except Exception as e:
                self.logger.error(f"关闭数据库连接失败: {e}")

    def _get_port(self) -> int:
        """获取服务端口号"""
        if self.settings:
            return self.settings.port
        return 5003

    def start(self):
        """启动 API 服务器"""
        port = self._get_port()
        self.logger.info(f"Starting {constants.SERVICE_NAME} on port {port}")
        uvicorn.run(self.app, host='0.0.0.0', port=port)
