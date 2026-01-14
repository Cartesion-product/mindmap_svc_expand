"""脑图生成 Agent (MindMap Agent)

根据接口设计实现完整的脑图生成流程：
1. get_md_content - 获取MD文档内容
2. process_md_images - 下载图像并转为base64
3. Document_Parser - 解析MD，按标题拆分内容
4. Skeleton_Planner - LLM定义Layer 1章节
5. Branch_Generator - 并行生成Layer 2和3
6. Tree_Assembler - 组装最终JSON
7. quality_check - 清洗与质检
8. upload_json - 上传JSON到MinIO
9. update_db - 更新数据库
"""
import time
import uuid
import json
import asyncio
import re
import base64
import aiohttp
from pathlib import Path
from typing import TypedDict, Optional, List, Dict, Any, Tuple
from langgraph.graph import StateGraph, END

from config.settings import get_settings
from common.enums import PaperTypeEnum
from services.minio_service import get_minio_service
from repositories.user_paper_repo import get_user_paper_repo
from repositories.system_paper_repo import get_system_paper_repo
from utilities.log_manager import get_celery_logger
from db.mongo import get_mongo_client
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

logger = get_celery_logger()
TEMP_DIR = Path(__file__).parent.parent / "data" / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


class MindMapState(TypedDict):
    result_id: str
    paper_id: str
    source: str
    paper_type: str
    agent_type: str
    user_id: str
    update_system: bool
    prompt: Optional[str]

    # Process states
    raw_md: Optional[str]
    processed_content: Optional[List[Dict]]  # 图像处理后的内容
    content_map: Optional[Dict[str, str]]  # 章节内容映射
    root_topic: Optional[str]  # 论文标题
    planned_sections: Optional[List[Dict]]  # L1章节规划
    sub_trees: Optional[List[Dict]]  # L2/L3分支
    final_json: Optional[Dict]  # 最终脑图JSON
    mindmap_json: Optional[Dict]  # 存储生成的脑图结构

    # Temporary resources
    temp_dir: Optional[str]  # 临时文件目录路径

    # Result
    file_path: Optional[str]  # MinIO 路径
    error_message: Optional[str]


class MindMapAgent:
    def __init__(self):
        self._settings = get_settings()
        self._minio_service = get_minio_service()
        self._user_repo = get_user_paper_repo()
        self._system_repo = get_system_paper_repo()
        self.llm = ChatOpenAI(
            model=self._settings.llm_model,
            temperature=0,
            base_url=self._settings.llm_base_url,
            api_key=self._settings.llm_api_key
        )
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()

    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(MindMapState)

        workflow.add_node("get_md_content", self.get_md_content_node)
        workflow.add_node("process_md_images", self.process_md_images_node)
        workflow.add_node("document_parser", self.document_parser_node)
        workflow.add_node("skeleton_planner", self.skeleton_planner_node)
        workflow.add_node("branch_generator", self.branch_generator_node)
        workflow.add_node("tree_assembler", self.tree_assembler_node)
        workflow.add_node("quality_check", self.quality_check_node)
        # workflow.add_node("upload_json", self.upload_json_node)
        workflow.add_node("update_db", self.update_db_node)

        workflow.set_entry_point("get_md_content")
        workflow.add_edge("get_md_content", "process_md_images")
        workflow.add_edge("process_md_images", "document_parser")
        workflow.add_edge("document_parser", "skeleton_planner")
        workflow.add_edge("skeleton_planner", "branch_generator")
        workflow.add_edge("branch_generator", "tree_assembler")
        workflow.add_edge("tree_assembler", "quality_check")
        # workflow.add_edge("quality_check", "upload_json")
        # workflow.add_edge("upload_json", "update_db")
        workflow.add_edge("quality_check", "update_db")
        workflow.add_edge("update_db", END)

        return workflow

    def get_md_content_node(self, state: MindMapState) -> MindMapState:
        """获取MD文档内容节点

        从 SV_KNOWLEDGE_DB.system_paper_original_content 表获取MD文档内容
        """
        try:
            mongo = get_mongo_client()
            doc = mongo.system_paper_content_collection.find_one({
                "paper_id": state["paper_id"],
                "source": state["source"]
            })
            if not doc or not doc.get("content"):
                raise ValueError(f"Content not found for paper_id={state['paper_id']}, source={state['source']}")
            state["raw_md"] = doc["content"]
            logger.info(f"[{state['result_id']}] 获取MD文档内容成功")
        except Exception as e:
            self._handle_error(state, str(e))
            raise e
        return state

    async def process_md_images_node(self, state: MindMapState) -> MindMapState:
        """处理MD图像节点

        下载MD中的外网图片并转为base64编码
        """
        try:
            raw_md = state["raw_md"]
            save_dir = TEMP_DIR / state["result_id"]
            processed_content, image_count = await self._process_md_images(raw_md, save_dir)

            state["processed_content"] = processed_content
            logger.info(f"[{state['result_id']}] 图像处理完成，处理了{image_count}个图像")
        except Exception as e:
            self._handle_error(state, f"图像处理失败: {str(e)}")
            raise e
        return state

    def document_parser_node(self, state: MindMapState) -> MindMapState:
        """文档解析节点

        将MD内容根据标题拆分，提取metadata作为Root Topic，拆分内容存储为content_map
        """
        try:
            processed_content = state["processed_content"]
            root_topic, content_map = self._parse_document(processed_content)

            state["root_topic"] = root_topic
            state["content_map"] = content_map
            logger.info(f"[{state['result_id']}] 文档解析完成，Root Topic: {root_topic}, 章节数: {len(content_map)}")
        except Exception as e:
            self._handle_error(state, f"文档解析失败: {str(e)}")
            raise e
        return state

    async def skeleton_planner_node(self, state: MindMapState) -> MindMapState:
        """骨架规划节点

        LLM拥有全局视角，只定义Layer 1，识别4-6个核心章节
        """
        try:
            content_map = state["content_map"]
            user_prompt = state.get("prompt", "")

            planned_sections = await self._plan_skeleton(content_map, user_prompt)
            state["planned_sections"] = planned_sections
            logger.info(f"[{state['result_id']}] 骨架规划完成，规划了{len(planned_sections)}个L1章节")
        except Exception as e:
            self._handle_error(state, f"骨架规划失败: {str(e)}")
            raise e
        return state

    async def branch_generator_node(self, state: MindMapState) -> MindMapState:
        """分支生成节点

        Map步骤：为每个L1节点生成其下的L2和L3
        """
        try:
            planned_sections = state["planned_sections"]
            content_map = state["content_map"]

            sub_trees = await self._generate_branches_parallel(planned_sections, content_map)
            state["sub_trees"] = sub_trees
            logger.info(f"[{state['result_id']}] 分支生成完成，生成了{len(sub_trees)}个子树")
        except Exception as e:
            self._handle_error(state, f"分支生成失败: {str(e)}")
            raise e
        return state

    def tree_assembler_node(self, state: MindMapState) -> MindMapState:
        """树组装节点

        Reduce步骤：将分散的零件组装成最终JSON
        """
        try:
            root_topic = state["root_topic"]
            planned_sections = state["planned_sections"]
            sub_trees = state["sub_trees"]

            final_json = self._assemble_tree(root_topic, planned_sections, sub_trees)
            state["final_json"] = final_json
            logger.info(f"[{state['result_id']}] 树组装完成")
        except Exception as e:
            self._handle_error(state, f"树组装失败: {str(e)}")
            raise e
        return state

    def quality_check_node(self, state: MindMapState) -> MindMapState:
        """质量检查节点（反思）

        清洗与质检：防止LLM幻觉，强制层级校准
        """
        try:
            final_json = state["final_json"]
            cleaned_json = self._quality_check(final_json)
            state["mindmap_json"] = cleaned_json
            logger.info(f"[{state['result_id']}] 质量检查完成")
        except Exception as e:
            self._handle_error(state, f"质量检查失败: {str(e)}")
            raise e
        return state

    # def upload_json_node(self, state: MindMapState) -> MindMapState:
    #     """上传 JSON 文件到 MinIO (备份)"""
    #     try:
    #         data = state["mindmap_json"]
    #         file_name = f"{state['result_id']}.json"
    #         local_path = TEMP_DIR / file_name
    #
    #         with open(local_path, "w", encoding="utf-8") as f:
    #             json.dump(data, f, ensure_ascii=False, indent=2)
    #
    #         # 使用 MinIO Service 上传
    #         # 注意：此处假设 minio_service 能够处理通用文件上传，或者需要适配
    #         # 这里简单复用 upload_task_results 里的逻辑或手动上传
    #         bucket = self._settings.get_bucket_name("mindmap", state["paper_type"])
    #         object_name = f"mindmaps/{state['result_id']}/{file_name}"
    #
    #         self._minio_service.client.fput_object(bucket, object_name, str(local_path))
    #         state["file_path"] = f"{bucket}/{object_name}"
    #
    #         # 清理临时文件
    #         if local_path.exists():
    #             local_path.unlink()
    #
    #         logger.info(f"[{state['result_id']}] JSON已上传至MinIO: {state['file_path']}")
    #
    #     except Exception as e:
    #         self._handle_error(state, f"上传失败: {str(e)}")
    #         raise e
    #     return state

    def update_db_node(self, state: MindMapState) -> MindMapState:
        """更新数据库 (User 和 System)"""
        try:
            # 1. 更新当前用户任务
            task = self._user_repo.find_by_result_id(state["result_id"])
            if task:
                task.mark_success(
                    file_path=state["file_path"],
                    result=json.dumps(state["mindmap_json"], ensure_ascii=False)  # 存入 JSON字符串
                )
                self._user_repo.update_task(task)

            # 2. 如果是系统任务，批量更新其他等待/运行中的任务，并更新系统表
            if state["update_system"]:
                # 更新系统表
                self._system_repo.update_file_path(
                    paper_id=state["paper_id"],
                    source=state["source"],
                    agent_type=state["agent_type"],
                    file_path=state["file_path"],
                    result_id=state["result_id"],
                    result=json.dumps(state["mindmap_json"], ensure_ascii=False)  # 存入 JSON字符串
                )

                # 批量更新其他用户的任务
                self._user_repo.update_running_tasks(
                    paper_id=state["paper_id"],
                    source=state["source"],
                    agent_type=state["agent_type"],
                    file_path=state["file_path"],
                    result=json.dumps(state["mindmap_json"], ensure_ascii=False)
                )
                logger.info(f"[{state['result_id']}] 系统记录同步更新完成")

        except Exception as e:
            self._handle_error(state, f"DB更新失败: {str(e)}")
            raise e
        finally:
            self._cleanup_temp_files(state)
        return state

    def _cleanup_temp_files(self, state):
        """清理临时文件"""
        try:
            temp_dir = TEMP_DIR / state.get("result_id", "")
            if temp_dir.exists() and temp_dir.is_dir():
                import shutil
                shutil.rmtree(temp_dir)
                logger.info(f"[{state.get('result_id')}] 清理临时文件: {temp_dir}")
        except Exception as e:
            logger.warning(f"[{state.get('result_id')}] 清理临时文件失败: {e}")

    def _handle_error(self, state, error_msg):
        logger.error(f"[{state.get('result_id')}] Agent Error: {error_msg}")
        task = self._user_repo.find_by_result_id(state["result_id"])
        if task:
            task.mark_failed(error_msg)
            self._user_repo.update_task(task)

        # 如果是系统任务且失败，清除系统锁（即删除空记录），允许重试
        if state.get("update_system") and state.get("paper_type") == PaperTypeEnum.SYSTEM.value:
            self._system_repo.delete_by_paper_id(state["paper_id"], state["agent_type"], state["source"])

        self._cleanup_temp_files(state)

    async def run(self, **kwargs):
        initial_state = MindMapState(
            result_id=kwargs["result_id"],
            paper_id=kwargs["paper_id"],
            source=kwargs["source"],
            paper_type=kwargs["paper_type"],
            agent_type=kwargs["agent_type"],
            user_id=kwargs["user_id"],
            update_system=kwargs["update_system"],
            prompt=kwargs.get("prompt"),
            raw_md=None,
            processed_content=None,
            content_map=None,
            root_topic=None,
            planned_sections=None,
            sub_trees=None,
            final_json=None,
            mindmap_json=None,
            file_path=None,
            error_message=None
        )
        return await self.app.ainvoke(initial_state)

    # ==================== Helper Methods ====================

    async def _process_md_images(self, md_content: str, save_dir: Path) -> Tuple[List[Dict], int]:
        """处理MD中的图像，下载并转为base64"""
        return await _download_images_from_md(md_content, save_dir)

    def _parse_document(self, processed_content: List[Dict]) -> Tuple[str, Dict[str, str]]:
        """解析文档，提取标题和章节内容"""
        # 将processed_content合并为纯文本用于解析
        full_text = ""
        for item in processed_content:
            if item["type"] == "text":
                full_text += item["text"]
            elif item["type"] == "image_url":
                # 保留图像占位符
                full_text += f"[IMAGE: {item['image_url']['url'][:50]}...]"

        # 提取论文标题（通常是第一个#标题）
        lines = full_text.split('\n')
        root_topic = "Research Paper"  # 默认标题
        for line in lines[:10]:  # 检查前10行
            if line.strip().startswith('# '):
                root_topic = line.strip()[2:].strip()
                break

        # 按标题拆分内容
        content_map = {}
        current_section = "Introduction"
        current_content = []

        for line in lines:
            line = line.strip()
            if line.startswith('# '):
                # 保存上一章节
                if current_content:
                    content_map[current_section] = '\n'.join(current_content)
                # 新章节
                current_section = line[2:].strip()
                current_content = []
            elif line.startswith('## ') or line.startswith('### '):
                # 子章节也作为独立章节
                if current_content:
                    content_map[current_section] = '\n'.join(current_content)
                current_section = line.lstrip('#').strip()
                current_content = [line]
            else:
                current_content.append(line)

        # 保存最后一个章节
        if current_content:
            content_map[current_section] = '\n'.join(current_content)

        return root_topic, content_map

    async def _plan_skeleton(self, content_map: Dict[str, str], user_prompt: str) -> List[Dict]:
        """使用LLM规划L1骨架"""
        content_summary = "\n".join([f"- {k}: {v[:200]}..." for k, v in content_map.items()])

        prompt = f"""
        你是一个学术论文分析专家。请分析这篇论文的主要结构，定义Layer 1的章节。

        要求：
        1. 只定义4-6个核心章节（如：背景、方法、实验、结论等）
        2. 每个章节包含：section_id（唯一标识）, section_topic（章节名称）
        3. 返回纯JSON数组格式，不要返回任何其他内容。

        用户要求：{user_prompt if user_prompt else "无"}

        论文内容摘要：
        {content_summary}

        返回格式：
        [
            {{"section_id": "background", "section_topic": "研究背景"}},
            {{"section_id": "method", "section_topic": "研究方法"}}
        ]
        """
        # 调用LLM生成
        messages = [
            # SystemMessage(content=SIMULATION_CHART_GENERATION_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]

        response = await self.llm.ainvoke(messages)
        content = response.content if hasattr(response, 'content') else str(response)
        json_str = content.replace("```json", "").replace("```", "").strip()

        return json.loads(json_str)

    async def _generate_branches_parallel(self, planned_sections: List[Dict], content_map: Dict[str, str]) -> List[Dict]:
        """并行生成分支（L2和L3）"""
        # 首先通过LLM匹配planned_sections和content_map的标题
        # 构建匹配提示词
        planned_items = [{"section_id": section["section_id"], "section_topic": section["section_topic"]} for section in
                         planned_sections]
        available_content_topics = list(content_map.keys())

        match_prompt = f"""
                你是一个内容匹配专家。请将规划的章节与实际的内容段落标题进行匹配。

                规划的章节列表（包含ID和主题）：
                {planned_items}

                可用的内容段落标题列表：
                {available_content_topics}

                请为每个规划的章节找到最匹配的内容段落标题。可以匹配1-3个相关的内容段落标题。
                如果找不到匹配项，返回空数组。

                严格按照以下JSON格式返回：
                [
                    {{
                        "section_id": "规划的章节ID",
                        "section_topic": "规划的章节主题",
                        "matched_content_topics": ["匹配的内容段落标题1", "匹配的内容段落标题2"]
                    }}
                ]

                重要：只返回JSON，不要返回任何其他内容。
                """

        messages = [
            HumanMessage(content=match_prompt)
        ]

        response = await self.llm.ainvoke(messages)
        content = response.content if hasattr(response, 'content') else str(response)

        # 提取JSON部分
        import re
        json_match = re.search(r'(\[.*\])', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            json_str = content.strip()

        try:
            matches = json.loads(json_str)
        except json.JSONDecodeError:
            # 如果解析失败，使用原始的匹配逻辑
            logger.warning("章节匹配失败，使用默认匹配逻辑")
            matches = []
            for section in planned_sections:
                section_topic = section["section_topic"]
                # 尝试精确匹配
                matched_topics = []
                if section_topic in content_map:
                    matched_topics = [section_topic]
                else:
                    # 尝试模糊匹配，最多匹配3个
                    for key in content_map.keys():
                        if section_topic.lower() in key.lower() or key.lower() in section_topic.lower():
                            matched_topics.append(key)
                            if len(matched_topics) >= 3:  # 最多3个
                                break
                    # 如果没找到模糊匹配，使用前3个内容标题
                    if not matched_topics:
                        matched_topics = list(content_map.keys())[:3]
                matches.append({
                    "section_id": section["section_id"],
                    "section_topic": section["section_topic"],
                    "matched_content_topics": matched_topics
                })

        # 创建匹配映射
        section_id_to_contents = {match["section_id"]: match["matched_content_topics"] for match in matches}

        tasks = []
        for section in planned_sections:
            section_id = section["section_id"]
            section_topic = section["section_topic"]
            # 使用匹配的标题获取内容
            matched_topics = section_id_to_contents.get(section_id, [])
            chunk_text = ""
            if matched_topics:
                # 将匹配的多个内容段落合并
                matched_contents = [content_map[topic] for topic in matched_topics if topic in content_map]
                chunk_text = "\n".join(matched_contents)

            # 如果仍然没有找到匹配的内容，使用所有内容的前几段
            if not chunk_text:
                logger.warning(f"未找到章节 '{section_topic}' 的匹配内容，使用默认内容")
                chunk_text = " ".join(list(content_map.values())[:3])

            tasks.append(self._generate_single_branch(section_id, section_topic, chunk_text))

        results = await asyncio.gather(*tasks)
        return [item for sublist in results for item in sublist]  # 展平结果

    async def _generate_single_branch(self, section_id: str, section_topic: str, chunk_text: str) -> List[Dict]:
        """生成单个分支的L2和L3"""
        prompt = f"""
        为论文章节"{section_topic}"生成思维导图的Layer 2和Layer 3结构。

        要求：
        1. Layer 2：3-5个子主题
        2. Layer 3：每个L2下1-3个具体内容点
        3. 每个节点包含：id（UUID）, topic（标题）, layer（层级）, note（备注）
        4. 返回该章节的完整子树结构

        章节内容：
        {chunk_text[:3000]}

        返回JSON格式：
        {{
            "section_id": "{section_id}",
            "sub_tree": {{
                "id": "uuid",
                "topic": "{section_topic}",
                "layer": 1,
                "note": "章节概述",
                "children": [
                    {{
                        "id": "uuid",
                        "topic": "L2主题1",
                        "layer": 2,
                        "note": "说明",
                        "children": [
                            {{
                                "id": "uuid",
                                "topic": "L3内容1",
                                "layer": 3,
                                "note": "具体内容"
                            }}
                        ]
                    }}
                ]
            }}
        }}
        
        重要：只返回JSON，不要返回任何其他内容。
        """

        messages = [
            # SystemMessage(content=SIMULATION_CHART_GENERATION_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]

        response = await self.llm.ainvoke(messages)

        content = response.content if hasattr(response, 'content') else str(response)
        json_str = content.replace("```json", "").replace("```", "").strip()

        result = json.loads(json_str)
        return [result]

    def _assemble_tree(self, root_topic: str, planned_sections: List[Dict], sub_trees: List[Dict]) -> Dict:
        """组装最终的树结构"""
        root = {
            "id": str(uuid.uuid4()),
            "topic": root_topic,
            "layer": 0,
            "note": "论文主题",
            "children": []
        }

        sub_tree_map = {tree["section_id"]: tree["sub_tree"] for tree in sub_trees}

        for section in planned_sections:
            section_id = section["section_id"]
            section_topic = section["section_topic"]

            l1_node = {
                "id": str(uuid.uuid4()),
                "topic": section_topic,
                "layer": 1,
                "note": f"章节：{section_topic}",
                "children": []
            }

            if section_id in sub_tree_map:
                sub_tree = sub_tree_map[section_id]
                if "children" in sub_tree:
                    l1_node["children"] = sub_tree["children"]

            root["children"].append(l1_node)

        return root

    def _quality_check(self, mindmap_json: Dict) -> Dict:
        """质量检查和清洗"""
        def traverse_and_clean(node: Dict, parent_layer: int = -1) -> Dict:
            """递归遍历并清洗节点"""
            # Rule 2: 检查是否有id，没有则补全
            if "id" not in node:
                node["id"] = str(uuid.uuid4())

            # Rule 3: 强制重写layer字段
            expected_layer = parent_layer + 1
            node["layer"] = expected_layer

            # Rule 1: 如果layer > 3，剪枝
            if expected_layer > 3:
                node["children"] = []
                return node

            # 递归处理子节点
            if "children" in node and isinstance(node["children"], list):
                cleaned_children = []
                for child in node["children"]:
                    if isinstance(child, dict):
                        cleaned_children.append(traverse_and_clean(child, expected_layer))
                node["children"] = cleaned_children

                return node

        return traverse_and_clean(mindmap_json)


# ==================== Helper Functions ====================

async def _download_images_from_md(md_content: str, save_dir: Path) -> Tuple[List[Dict], int]:
    """从MD内容中提取外网图片URL并下载到本地，返回处理后的内容和图像数量"""
    save_dir.mkdir(parents=True, exist_ok=True)

    # 匹配Markdown图片语法中的URL
    pattern = r'!\[.*?\]\((https?://[^\s\)]+\.(?:jpg|jpeg|png|gif|bmp|webp|tiff|tif)[^\)]*)\)'
    matches = list(re.finditer(pattern, md_content, re.IGNORECASE))

    url_to_local = {}
    unique_urls = []

    # 收集唯一URL
    for match in matches:
        url = match.group(1)
        if url not in url_to_local and url not in unique_urls:
            unique_urls.append(url)

    # 下载图像
    if unique_urls:
        async with aiohttp.ClientSession() as session:
            tasks = []
            for idx, url in enumerate(unique_urls):
                tasks.append(_download_single_image(session, url, save_dir, idx))

            results = await asyncio.gather(*tasks)
            for url, local_path in zip(unique_urls, results):
                if local_path:
                    url_to_local[url] = local_path

    # 处理内容，将图像替换为base64
    content_parts = []
    last_pos = 0
    image_count = 0

    for match in matches:
        # 添加匹配前的文本
        if match.start() > last_pos:
            text_part = md_content[last_pos:match.start()]
            if text_part.strip():
                content_parts.append({
                    "type": "text",
                    "text": text_part
                })

        # 处理图像
        image_url = match.group(1)
        local_image_path = url_to_local.get(image_url)

        if local_image_path and Path(local_image_path).exists():
            base64_str = _encode_image_to_base64(local_image_path)
            if base64_str:
                mime_type = _get_image_mime_type(local_image_path)
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_str}"
                    }
                })
                image_count += 1
            else:
                # 编码失败，保留原文
                content_parts.append({
                    "type": "text",
                    "text": match.group(0)
                })
        else:
            # 下载失败，保留原文
            content_parts.append({
                "type": "text",
                "text": match.group(0)
            })

        last_pos = match.end()

    # 添加剩余文本
    if last_pos < len(md_content):
        remaining_text = md_content[last_pos:]
        if remaining_text.strip():
            content_parts.append({
                "type": "text",
                "text": remaining_text
            })

    return content_parts, image_count


async def _download_single_image(session: aiohttp.ClientSession, url: str, save_dir: Path, idx: int) -> Optional[str]:
    """下载单个图像"""
    try:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                content = await response.read()
                # 推断文件扩展名
                ext = _guess_image_extension(url, content)
                if not ext:
                    return None

                file_path = save_dir / f"image_{idx}.{ext}"
                with open(file_path, "wb") as f:
                    f.write(content)

                return str(file_path)
    except Exception as e:
        logger.warning(f"Failed to download image {url}: {e}")

    return None


def _encode_image_to_base64(image_path: str) -> Optional[str]:
    """将图像编码为base64"""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        return None


def _get_image_mime_type(image_path: str) -> str:
    """获取图像的MIME类型"""
    ext = Path(image_path).suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp',
        '.tiff': 'image/tiff',
        '.tif': 'image/tiff'
    }
    return mime_types.get(ext, 'image/jpeg')


def _guess_image_extension(url: str, content: bytes) -> Optional[str]:
    """推断图像文件扩展名"""
    # 先从URL推断
    url_ext = Path(url).suffix.lower()
    if url_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif']:
        return url_ext[1:]  # 去掉点

    # 从文件头推断
    if content.startswith(b'\xff\xd8'):
        return 'jpg'
    elif content.startswith(b'\x89PNG'):
        return 'png'
    elif content.startswith(b'GIF8'):
        return 'gif'
    elif content.startswith(b'BM'):
        return 'bmp'
    elif content.startswith(b'RIFF') and b'WEBP' in content[:20]:
        return 'webp'

    return None


# 便捷调用函数
async def run_mindmap_agent(**kwargs):
    agent = MindMapAgent()
    return await agent.run(**kwargs)
