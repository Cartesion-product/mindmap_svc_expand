## 一、接口设计

### 1、任务创建接口

**接口路径**: `POST /api/v1/mindmap/tasks/create`

**功能描述**: 创建PPT/Poster生成任务，添加至队列，返回任务ID

**请求参数**:

```json
{
  "paper_id": "9407014",		// 论文ID
  "source": "arxiv",			// 论文数据源 arxiv、acl、neurips
  "title": "脑图智绘",		     // 默认 脑图智绘
  "paper_type": "system",  		// system-系统论文（默认）, user-用户论文
  "prompt": "用户额外提示词",		// prompt 用户入参
}
```

**响应结果**:

```json
{
  "code": 200,
  "message": "任务创建成功",
  "data": {
      "task_id": ""
  }
}
```

**业务逻辑**:

1. 验证必填参数（paper_id, source），并通过token_payload: dict[str, None] = Depends(token_decoder)参数获取user_id = token_payload['user_id']；
2. 检查任务队列（可执行任务上限2，等待队列上限5），若等待队列上限则返回-任务繁忙；
3. 创建任务记录，状态为"等待中(waiting)"，写入 user_paper_agent_result 库记录，初始化状态init_state；
4. 根据 paper_id, source, agent_type（mindmap） 查询 system_paper_agent_result 表中是否存在记录。设置默认标记False。
   - 有记录的默认结果是否有值，
     - 有值则获取值更新创建的任务，并更新状态为success，返回任务ID；
     - 无值则将状态改为running，返回任务ID；
   - 若是无记录则创建系统记录，并设置标记True，后续用于系统数据更新节点，提交Celery异步任务（agent节点流程），返回任务ID；

celery异步任务就正常执行agent。



**Agent节点流程**（若是整个流程报错抛异常，则更新任务失败原因和时间）：

1. get_md_content。根据source和paper_id获取 SV_KNOWLEDGE_DB库中**system_paper_original_content**表的MD文档内容；
2. process_md_images。将md文件中的图像下载到本地，如代码：_download_images_from_md，然后将图像转为base64替换原文内容，_replace_images_with_base64；
3. `Document_Parser` (Python Logic)。将md内容根据标题拆分，提取 metadata（论文标题）作为 Root Topic，拆分内容存储为content_map；
4. `Skeleton_Planner` (LLM)。拥有全局视角的LLM，**只定义 Layer 1**，不关注细节，识别 4-6 个核心章节（如：背景、方法、实验、结论）；
5.  `Branch_Generator` (LLM - Parallel)。这步**Map** 步骤。为每个 L1 节点生成其下的 L2 和 L3。输入：`section_topic` (来自 Planner) + `chunk_text` (该章节原文)，章节原文从content_map获取；输出：将生成的 List append 到 `sub_trees` 状态中；
6. `Tree_Assembler` (Python Logic)。**Reduce** 步骤。将分散的零件组装成车。a、初始化 `root` 节点对象（Layer 0）。b、遍历 `planned_sections`，创建 Layer 1 节点挂载到 root。c、根据 `section_id` 匹配，将 Node 3 生成的 `sub_trees` 挂载到对应的 Layer 1 节点下； 输出初步的 `final_json`
7. Python Logic - 兜底保障。**清洗与质检**。防止 LLM 这里那里多生成了一层。递归遍历整棵树。输出：Cleaned `final_json`
   - **Rule 1 (剪枝)**：如果发现任何节点的 `layer > 3`，强行删除其 `children`，将其变成叶子节点。
   - **Rule 2 (ID修复)**：检查所有节点是否有 `id`，如果没有则补全 UUID。
   - **Rule 3 (层级校准)**：强制重写 `layer` 字段，确保 `root`=0, `children`=1, `grand_children`=2... 避免 LLM 幻觉标错层级。
8. 任务结果保存状态更新。





```python
async def _download_images_from_md(md_content: str, save_dir: Path) -> Dict[str, str]:
    """从MD内容中提取外网图片URL并下载到本地，返回URL到本地路径的映射"""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 匹配Markdown图片语法中的URL (http/https)
    pattern = r'!\[.*?\]\((https?://[^\s\)]+\.(?:jpg|jpeg|png|gif|bmp|webp|tiff|tif)[^\)]*)\)'
    matches = list(re.finditer(pattern, md_content, re.IGNORECASE))
    
    if not matches:
        logger.info("No external images found in markdown")
        return {}
    
    logger.info(f"Found {len(matches)} external images to download")
    url_to_local = {}
    unique_urls = []
    
    # 收集唯一URL
    for match in matches:
        url = match.group(1)
        if url not in url_to_local and url not in unique_urls:
            unique_urls.append(url)
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for idx, url in enumerate(unique_urls):
            tasks.append(_download_image(session, url, save_dir, idx))
        
        results = await asyncio.gather(*tasks)
        for url, local_path in zip(unique_urls, results):
            if local_path:
                url_to_local[url] = local_path
    
    logger.info(f"Successfully downloaded {len(url_to_local)} images")
    return url_to_local
    
   
   
def _replace_images_with_base64(markdown_content: str, url_to_local: Dict[str, str],) -> Tuple[List, int]:
    """
    将 markdown 中的图像引用替换为 base64 编码图像，保持位置
    
    Args:
        markdown_content: Markdown text content
        url_to_local: images_base_map
    """
    content_parts = []
    last_pos = 0
    image_count = 0
    
    pattern = r'(!\[.*?\]\((.*?\.(?:jpg|jpeg|png|gif|bmp|webp|tiff|tif))\)|Image Path:\s*([^\r\n]*?\.(?:jpg|jpeg|png|gif|bmp|webp|tiff|tif)))'
    
    for match in re.finditer(pattern, markdown_content, re.IGNORECASE | re.DOTALL):
        # Add text before this image
        if match.start() > last_pos:
            text_part = markdown_content[last_pos:match.start()]
            if text_part.strip():
                content_parts.append({
                    "type": "text",
                    "text": text_part
                })
        
        # Extract image path (from either group 2 or 3)
        image_path = match.group(2) if match.group(2) else match.group(3)
        image_path = image_path.strip()
        
        local_image_path = url_to_local.get(image_path, image_path)
        
        # Try to encode image
        if Path(local_image_path).exists():
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
                logger.debug(f"Embedded image at position {match.start()}: {local_image_path}")
            else:
                # If encoding fails, keep original text
                content_parts.append({
                    "type": "text",
                    "text": match.group(0)
                })
        else:
            logger.warning(f"Image not found: {local_image_path}")
            # Keep original text reference
            content_parts.append({
                "type": "text",
                "text": match.group(0)
            })
        
        last_pos = match.end()
    
    # Add remaining text after last image
    if last_pos < len(markdown_content):
        remaining_text = markdown_content[last_pos:]
        if remaining_text.strip():
            content_parts.append({
                "type": "text",
                "text": remaining_text
            })
    
    return content_parts, image_count
```



### 2、任务删除接口

**接口路径**: `DELETE /api/v1/mindmap/tasks/{task_id}/delete`

**功能描述**: 删除任务，运行中任务停止执行

**请求参数**: 

- `task_id`: 任务ID（路径参数）

**响应结果**:

```json
{
  "code": 200,
  "message": "任务删除成功"
}
```

**业务逻辑**:

1. 根据task_id查询任务；
2. 如任务状态为"运行中(10)"，调用Celery revoke停止任务；
3. 删除用户数据库任务记录；
4. 不影响系统论文的默认结果（system_paper_agent_result表）；



### 3、任务详情接口

**接口路径**: `GET /api/v1/mindmap/tasks/{task_id}/detail`

**功能描述**: 查询任务详情；

**请求参数**:

- `task_id`: 任务ID（路径参数）

**响应结果**:

```json
{
  "code": 200,
  "message": "查询成功",
  "data": {
    "task_id": "任务ID",
    ...
  }
}
```

**业务逻辑**:

1. 根据task_id查询任务；
2. 将查询的任务数据复制resp并返回；
