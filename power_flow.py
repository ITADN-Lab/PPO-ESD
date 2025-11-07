import numpy as np
from config import Sbase


def backward_forward_sweep(bus_data, branch_data, max_iter=100, tol=1e-6):
    """
    前推回代法计算潮流

    Args:
        bus_data: 节点数据
        branch_data: 支路数据
        max_iter: 最大迭代次数
        tol: 收敛容差

    Returns:
        tuple: (节点电压, 支路电流)
    """
    nodes = sorted(bus_data.keys())
    V = {bus: complex(1.0, 0.0) for bus in nodes}
    I = {i: complex(0.0, 0.0) for i in range(len(branch_data))}

    for _ in range(max_iter):
        V_prev = V.copy()

        # 回代过程
        for i in reversed(range(len(branch_data))):
            line = branch_data[i]
            to_bus = line['to']
            S_pu = complex(bus_data[to_bus]['P_pu'], bus_data[to_bus]['Q_pu'])
            V_to = V[to_bus]
            if abs(V_to) < 1e-6:
                V_to = complex(1e-6, 0.0)
            I_load = np.conj(S_pu / V_to)
            child_lines = [idx for idx, l in enumerate(branch_data) if l['from'] == to_bus]
            I[i] = I_load + sum(I[child] for child in child_lines)

        # 前推过程
        for i in range(len(branch_data)):
            line = branch_data[i]
            from_bus = line['from']
            to_bus = line['to']
            V[to_bus] = V[from_bus] - I[i] * line['Z_pu']

        max_diff = max(abs(V[bus] - V_prev[bus]) for bus in nodes)
        if max_diff < tol:
            break

    return V, I


def calculate_losses(branch_data, I):
    """
    计算线路损耗

    Args:
        branch_data: 支路数据
        I: 支路电流

    Returns:
        tuple: (各支路损耗, 总损耗)
    """
    line_losses = []
    total_loss = 0.0
    for idx, line in enumerate(branch_data):
        I_mag = abs(I[idx])
        loss_pu = (I_mag ** 2) * line['R_pu']
        loss_MW = loss_pu * Sbase
        line_losses.append(loss_MW)
        total_loss += loss_MW
    return line_losses, total_loss