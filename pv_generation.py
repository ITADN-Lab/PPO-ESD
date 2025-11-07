import numpy as np
from config import Sbase


def generate_multi_pv_output(pv_nodes_config):
    """
    生成多节点光伏输出

    Args:
        pv_nodes_config: 光伏节点配置字典 {节点号: 容量比例}

    Returns:
        dict: 各节点的光伏输出（标幺值）
    """
    base_pv_curve = np.array([0, 0, 0, 0, 50, 80, 90, 100, 150, 400, 500, 600,
                              700, 700, 650, 500, 400, 100, 50, 0, 0, 0, 0, 0])

    pv_outputs = {}
    for node, capacity_ratio in pv_nodes_config.items():
        node_pv_output = capacity_ratio * base_pv_curve
        pv_outputs[node] = node_pv_output / 1000.0 / Sbase

    return pv_outputs