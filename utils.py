import os
import numpy as np
from config import SAVE_DIR

def save_loss_data(hours, losses, filename):
    """保存损耗数据"""
    filepath = os.path.join(SAVE_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("小时\t线损(kW)\n")
        for hour, loss in enumerate(losses):
            f.write(f"{hour}\t{loss:.4f}\n")
    return filepath

def save_loss_ratio_data(hours, loss_ratios, filename):
    """保存线损率数据"""
    filepath = os.path.join(SAVE_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("小时\t线损率(%)\n")
        for hour, loss_ratio in enumerate(loss_ratios):
            f.write(f"{hour}\t{loss_ratio:.4f}\n")
    return filepath

def calculate_loss_statistics(losses_without_storage, losses_with_storage, demand_without_storage):
    """计算损耗统计信息"""
    all_total_loss_no_storage = sum(losses_without_storage)
    all_total_loss_with_storage = sum(losses_with_storage)
    loss_reduction = all_total_loss_no_storage - all_total_loss_with_storage
    loss_reduction_percent = (loss_reduction / all_total_loss_no_storage) * 100 if all_total_loss_no_storage > 0 else 0

    total_demand_no_storage = sum(demand_without_storage)
    overall_loss_ratio_no_storage = (all_total_loss_no_storage / total_demand_no_storage) * 100 if total_demand_no_storage > 0 else 0
    overall_loss_ratio_with_storage = (all_total_loss_with_storage / total_demand_no_storage) * 100 if total_demand_no_storage > 0 else 0

    return {
        'total_loss_no_storage': all_total_loss_no_storage,
        'total_loss_with_storage': all_total_loss_with_storage,
        'loss_reduction': loss_reduction,
        'loss_reduction_percent': loss_reduction_percent,
        'overall_loss_ratio_no_storage': overall_loss_ratio_no_storage,
        'overall_loss_ratio_with_storage': overall_loss_ratio_with_storage
    }