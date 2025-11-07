import os
import numpy as np
import matplotlib.pyplot as plt
from config import SAVE_DIR, PLOT_CONFIG, bus_lists
from utils import save_loss_data, save_loss_ratio_data, calculate_loss_statistics
from matplotlib import rcParams


def setup_plotting():
    """设置绘图参数"""
    rcParams.update(PLOT_CONFIG)
    os.makedirs(SAVE_DIR, exist_ok=True)


def plot_pv_profiles(env, pv_config, save_dir=SAVE_DIR):
    """绘制光伏出力曲线"""
    colors = plt.cm.tab10.colors
    hours = range(24)

    plt.figure()
    for node, ratio in pv_config.items():
        pv_output_kw = env.pv_outputs[node] * env.Sbase * 1000
        plt.plot(hours, pv_output_kw, linewidth=1.5,
                 label=f'Node {node} (Capacity ratio: {ratio})')

    plt.xlabel('Time (hour)', fontweight='bold')
    plt.ylabel('PV Output (kW)', fontweight='bold')
    plt.title('Photovoltaic Output Profiles at Different Nodes', fontweight='bold')
    plt.legend(frameon=True, fancybox=False, edgecolor='black')
    plt.xticks(range(0, 24, 2))
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.tight_layout(pad=0.5)
    plt.savefig(os.path.join(save_dir, "6IEEE33多节点光伏_光伏出力曲线.png"), dpi=300, bbox_inches='tight')
    plt.show()


def plot_ess_dispatch(best_storage_power_profile, save_dir=SAVE_DIR):
    """绘制储能调度策略"""
    colors = plt.cm.tab10.colors
    hours = range(24)

    plt.figure()
    charge_power = np.where(best_storage_power_profile < 0, best_storage_power_profile, 0)
    discharge_power = np.where(best_storage_power_profile > 0, best_storage_power_profile, 0)

    plt.bar(hours, discharge_power, color=colors[0], alpha=0.8, label='Discharge')
    plt.bar(hours, charge_power, color=colors[1], alpha=0.8, label='Charge')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.xlabel('Time (hour)', fontweight='bold')
    plt.ylabel('ESS Power (kW)', fontweight='bold')
    plt.title('Optimal ESS Dispatch Strategy(Multi-Node PV Access)', fontweight='bold')
    plt.legend(frameon=True, fancybox=False, edgecolor='black',
               loc='lower right', bbox_to_anchor=(0.99, 0.01), fontsize=15)
    plt.xticks(range(0, 24, 1))
    plt.grid(True, alpha=0.2, linestyle='--', axis='y')
    plt.tight_layout(pad=0.5)
    plt.savefig(os.path.join(save_dir, "6IEEE33多节点光伏_最优储能调度策略.png"), dpi=300, bbox_inches='tight')
    plt.show()


def plot_line_loss_comparison(env, best_storage_power_profile, save_dir=SAVE_DIR):
    """绘制线损对比图"""
    colors = plt.cm.tab10.colors
    hours = range(24)

    # 计算有/无储能的线损
    losses_without_storage_with_pv = []
    best_losses_with_storage = []
    demand_without_storage = []

    for hour in range(24):
        total_loss_no_storage = env.calculate_losses_without_storage_with_pv(hour)
        losses_without_storage_with_pv.append(total_loss_no_storage * 1e3)

        # 使用 env.bus_lists 而不是 env.bus_lists
        current_bus = env.bus_lists[hour]  # 现在 env 有这个属性了
        demand_no_storage = sum(current_bus[:, 1])
        demand_without_storage.append(demand_no_storage)

        storage_power = best_storage_power_profile[hour] / (env.Sbase * 1000)
        voltages, line_losses, total_loss_with_storage = env.calculate_power_flow(hour, storage_power)
        best_losses_with_storage.append(total_loss_with_storage * 1e3)

    # 保存数据
    save_loss_data(hours, best_losses_with_storage,
                   "6负载系数1.0,光伏系数2.00,IEEE33多节点光伏_最优储能调度线损数据.txt")
    save_loss_data(hours, losses_without_storage_with_pv,
                   "6未接储能负载系数1.0,光伏系数2.00,IEEE33多节点光伏_最优储能调度线损数据.txt")

    # 绘制线损对比图
    plt.figure()
    plt.plot(hours, best_losses_with_storage, 'o-', linewidth=1.5,
             markersize=4, label='PPO-ESD', color=colors[0])
    plt.plot(hours, losses_without_storage_with_pv, 's--', linewidth=1.5,
             markersize=4, label='Without PPO-ESD', color=colors[1])

    plt.xlabel('Time (hour)', fontweight='bold')
    plt.ylabel('Line Loss (kW)', fontweight='bold')
    plt.title('Comparison of Line Loss(Multi-Node PV Access)', fontweight='bold')
    plt.legend(frameon=True, fancybox=False, edgecolor='black')
    plt.xticks(range(0, 24, 2))
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.tight_layout(pad=0.5)
    plt.savefig(os.path.join(save_dir, "6IEEE33多节点光伏_总线损耗对比图.png"), dpi=300, bbox_inches='tight')
    plt.show()

    return losses_without_storage_with_pv, best_losses_with_storage, demand_without_storage


def plot_loss_rate_comparison(env, best_storage_power_profile, save_dir=SAVE_DIR):
    """绘制线损率对比图"""
    colors = plt.cm.tab10.colors
    hours = range(24)

    # 计算线损率
    losses_without_storage_with_pv = []
    best_losses_with_storage = []
    demand_without_storage = []

    for hour in range(24):
        total_loss_no_storage = env.calculate_losses_without_storage_with_pv(hour)
        losses_without_storage_with_pv.append(total_loss_no_storage * 1e3)

        # 使用 env.bus_lists
        current_bus = env.bus_lists[hour]
        demand_no_storage = sum(current_bus[:, 1])
        demand_without_storage.append(demand_no_storage)

        storage_power = best_storage_power_profile[hour] / (env.Sbase * 1000)
        voltages, line_losses, total_loss_with_storage = env.calculate_power_flow(hour, storage_power)
        best_losses_with_storage.append(total_loss_with_storage * 1e3)

    # 计算线损率
    loss_ratios_without_storage = [loss / demand * 100 if demand > 0 else 0 for loss, demand in
                                   zip(losses_without_storage_with_pv, demand_without_storage)]
    loss_ratios_with_storage = [loss / demand * 100 if demand > 0 else 0 for loss, demand in
                                zip(best_losses_with_storage, demand_without_storage)]

    # 保存线损率数据
    save_loss_ratio_data(hours, loss_ratios_with_storage,
                         "6负载系数1.0,光伏系数2.00,IEEE33多节点光伏_最优储能调度线损率数据.txt")
    save_loss_ratio_data(hours, loss_ratios_without_storage,
                         "6未接储能负载系数1.0,光伏系数2.00,IEEE33多节点光伏_最优储能调度线损率数据.txt")

    # 绘制线损率对比图
    plt.figure()
    plt.plot(hours, loss_ratios_with_storage, 'o-', linewidth=1.5,
             markersize=4, label='PPO-ESD', color=colors[0])
    plt.plot(hours, loss_ratios_without_storage, 's--', linewidth=1.5,
             markersize=4, label='Without PPO-ESD', color=colors[1])

    plt.xlabel('Time (hour)', fontweight='bold')
    plt.ylabel('Line Loss Rate (%)', fontweight='bold')
    plt.title('Comparison of Line Loss Rate(Multi-Node PV Access)', fontweight='bold')
    plt.legend(frameon=True, fancybox=False, edgecolor='black')
    plt.xticks(range(0, 24, 2))
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.tight_layout(pad=0.5)
    plt.savefig(os.path.join(save_dir, "6IEEE33多节点光伏_线损率对比图.png"), dpi=300, bbox_inches='tight')
    plt.show()

    return losses_without_storage_with_pv, best_losses_with_storage, demand_without_storage


def plot_training_curves(timesteps, total_losses, total_rewards_list, save_dir=SAVE_DIR):
    """绘制训练过程曲线"""
    colors = plt.cm.tab10.colors

    # 1. 线损训练曲线
    plt.figure(figsize=(8, 4.5))
    plt.plot(timesteps, total_losses, color=colors[0], label='Line Loss')
    plt.xlabel('Training Days', fontweight='bold')
    plt.ylabel('Line Loss (kW)', fontweight='bold')
    plt.title('Total Line Loss(Multi-Node PV Access)', fontweight='bold')
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.legend(frameon=True, fancybox=False, edgecolor='black')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "6IEEE33多节点光伏训练线损曲线total_losses_24h_action.png"))
    plt.show()

    # 2. 奖励曲线
    plt.figure(figsize=(8, 4.5))
    plt.plot(timesteps, total_rewards_list, color=colors[1], label='Total Reward')
    plt.xlabel('Training Days', fontweight='bold')
    plt.ylabel('Reward Value', fontweight='bold')
    plt.title('Total Reward(Multi-Node PV Access)', fontweight='bold')
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.legend(frameon=True, fancybox=False, edgecolor='black')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "6IEEE33多节点光伏奖励曲线reward_and_penalty_curve.png"))
    plt.show()

    # 3. 线损与奖励双Y轴图
    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    ax1.plot(timesteps, total_losses, color=colors[0], label='Line Loss')
    ax1.set_xlabel('Training Days', fontweight='bold')
    ax1.set_ylabel('Power Loss (kW)', fontweight='bold', color=colors[0])
    ax1.tick_params(axis='y', labelcolor=colors[0])

    ax2 = ax1.twinx()
    ax2.plot(timesteps, total_rewards_list, color=colors[1], label='Reward')
    ax2.set_ylabel('Reward Value', fontweight='bold', color=colors[1])
    ax2.tick_params(axis='y', labelcolor=colors[1])

    plt.title('Line Loss and Reward(Multi-Node PV Access)', fontweight='bold')
    fig.tight_layout(pad=0.5)
    fig.savefig(os.path.join(save_dir, "6IEEE33多节点光伏线损与奖励曲线总图.png"))
    plt.show()


def print_storage_dispatch(best_storage_power_profile):
    """打印最优储能调度策略"""
    print(f"\n最优储能功率调度 (kW):")
    for hour in range(24):
        power = best_storage_power_profile[hour]
        if power > 0:
            print(f"第{hour:2d}小时: 放电 {power:6.2f} kW")
        elif power < 0:
            print(f"第{hour:2d}小时: 充电 {abs(power):6.2f} kW")
        else:
            print(f"第{hour:2d}小时: 无动作  0.00 kW")


def print_statistics(losses_without_storage, losses_with_storage, demand_without_storage,
                     total_charge_energy, total_discharge_energy, energy_imbalance):
    """打印统计信息"""
    stats = calculate_loss_statistics(losses_without_storage, losses_with_storage, demand_without_storage)

    print(f"\n多节点光伏系统损耗统计:")
    print(f"未接入储能全天损耗: {stats['total_loss_no_storage']:.2f} kW")
    print(f"光储协同全天损耗: {stats['total_loss_with_storage']:.2f} kW")
    print(f"损耗降低: {stats['loss_reduction']:.2f} kW ({stats['loss_reduction_percent']:.2f}%)")
    print(f"未接储能全天线损率: {stats['overall_loss_ratio_no_storage']:.2f}%")
    print(f"光储协同全天线损率: {stats['overall_loss_ratio_with_storage']:.2f}%)")
    print(f"全天充电量: {total_charge_energy:.2f} kWh")
    print(f"全天放电量: {total_discharge_energy:.2f} kWh")
    print(f"充放功率偏差: {energy_imbalance:.2f} kWh")

    return stats


def save_summary(stats, total_charge_energy, total_discharge_energy, energy_imbalance, save_dir=SAVE_DIR):
    """保存结果摘要"""
    summary_data = [
        ["Total Loss without ESS", f"{stats['total_loss_no_storage']:.2f} kW"],
        ["Total Loss with ESS", f"{stats['total_loss_with_storage']:.2f} kW"],
        ["Loss Reduction", f"{stats['loss_reduction']:.2f} kW ({stats['loss_reduction_percent']:.2f}%)"],
        ["Overall Loss Rate without ESS", f"{stats['overall_loss_ratio_no_storage']:.2f}%"],
        ["Overall Loss Rate with ESS", f"{stats['overall_loss_ratio_with_storage']:.2f}%"],
        ["Total Charging Energy", f"{total_charge_energy:.2f} kWh"],
        ["Total Discharging Energy", f"{total_discharge_energy:.2f} kWh"],
        ["Energy Imbalance", f"{energy_imbalance:.2f} kWh"]
    ]

    with open(os.path.join(save_dir, "6IEEE33多节点结果数据记录.txt"), "w", encoding='utf-8') as f:
        f.write("Performance Summary\n")
        f.write("=" * 40 + "\n")
        for row in summary_data:
            f.write(f"{row[0]:<35} {row[1]}\n")

    print(f"结果摘要已保存至: {os.path.join(save_dir, '6IEEE33多节点结果数据记录.txt')}")


def create_all_plots(env, pv_config, best_storage_power_profile,
                     total_charge_energy, total_discharge_energy, energy_imbalance,
                     total_losses, timesteps, total_rewards_list, save_dir=SAVE_DIR):
    """创建所有图表"""
    setup_plotting()

    # 1. 光伏出力曲线
    plot_pv_profiles(env, pv_config, save_dir)

    # 2. 储能调度策略
    plot_ess_dispatch(best_storage_power_profile, save_dir)
    print_storage_dispatch(best_storage_power_profile)

    # 3. 线损对比
    losses_without_storage, losses_with_storage, demand_without_storage = plot_line_loss_comparison(
        env, best_storage_power_profile, save_dir)

    # 4. 线损率对比
    plot_loss_rate_comparison(env, best_storage_power_profile, save_dir)

    # 5. 训练过程曲线
    plot_training_curves(timesteps, total_losses, total_rewards_list, save_dir)

    # 6. 打印统计信息
    stats = print_statistics(losses_without_storage, losses_with_storage, demand_without_storage,
                             total_charge_energy, total_discharge_energy, energy_imbalance)

    # 7. 保存结果摘要
    save_summary(stats, total_charge_energy, total_discharge_energy, energy_imbalance, save_dir)

    return stats