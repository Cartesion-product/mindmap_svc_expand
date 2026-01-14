"""MindMap 工具模块

提供脑图节点的数据结构、HTML生成等功能
"""
import json
from pyecharts import options as opts
from pyecharts.charts import Tree


class MindMapNode:
    """脑图节点类"""
    def __init__(self, id=None, topic="", layer=0, note="", children=None):
        self.id = id
        self.topic = topic
        self.layer = layer
        self.note = note
        self.children = children or []

    def to_dict(self):
        """将对象转换为字典格式"""
        return {
            "id": self.id,
            "topic": self.topic,
            "layer": self.layer,
            "note": self.note,
            "children": [child.to_dict() for child in self.children] if self.children else []
        }

    @classmethod
    def from_dict(cls, data):
        """从字典创建对象"""
        node = cls(
            id=data.get("id"),
            topic=data.get("topic"),
            layer=data.get("layer"),
            note=data.get("note"),
            children=[]
        )
        if "children" in data:
            node.children = [cls.from_dict(child) for child in data["children"]]
        return node


def build_echarts_data_from_object(node):
    """从MindMapNode对象构建ECharts数据"""
    data_item = {
        "name": node.topic,
        "value": node.note,
        "children": []
    }

    if node.children:
        for child in node.children:
            data_item["children"].append(build_echarts_data_from_object(child))

    return data_item


def get_leaf_count_from_object(node):
    """从MindMapNode对象计算叶子节点数"""
    if not node.children:
        return 1
    count = 0
    for child in node.children:
        count += get_leaf_count_from_object(child)
    return count


def create_html_mindmap_from_object(root_node, output_path: str = None):
    """使用MindMapNode对象创建HTML脑图

    Args:
        root_node: 根节点对象
        output_path: 输出HTML文件路径，如果为None则不保存文件，只返回HTML内容

    Returns:
        str: HTML内容（如果output_path为None）
        或保存文件路径（如果output_path指定）
    """
    from pathlib import Path

    leaf_count = get_leaf_count_from_object(root_node)
    per_node_height = 60
    calculated_height = max(600, leaf_count * per_node_height)
    canvas_height = f"{calculated_height}px"

    echarts_data = [build_echarts_data_from_object(root_node)]

    c = (
        Tree(
            init_opts=opts.InitOpts(
                width="1200px",
                height=canvas_height,
                page_title="论文结构脑图"
            )
        )
            .add(
            series_name="论文结构",
            data=echarts_data,
            orient="LR",
            pos_top="5%",
            pos_left="5%",
            pos_bottom="5%",
            pos_right="30%",
            symbol_size=8,
            initial_tree_depth=3,
            label_opts=opts.LabelOpts(
                position="right",
                vertical_align="middle",
                font_size=16,
                font_family="Microsoft YaHei"
            ),
            edge_shape='curve',
        )
            .set_global_opts(
            title_opts=opts.TitleOpts(title="脑图智绘"),
            tooltip_opts=opts.TooltipOpts(
                trigger="item",
                trigger_on="mousemove|click",
                formatter="{b} <br/> 备注: {c}"
            ),
        )
    )

    if output_path:
        c.render(output_path)
        return output_path
    else:
        return c.render_embed()
