import numpy as np
import gym
from stable_baselines3 import PPO
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import DEFAULT_PV_CONFIG
from power_grid_env import PowerGridEnv
from plotting import create_all_plots

def main():
    """主函数"""
    # 设置编码
    sys.stdout.reconfigure(encoding='utf-8')

    # 创建环境和模型
    env = PowerGridEnv(storage_node=18, pv_nodes_config=DEFAULT_PV_CONFIG,
                       discharge_limit=1000, charge_limit=1000)

    model = PPO("MlpPolicy", env,
                learning_rate=3e-3,
                n_steps=96,
                batch_size=48,
                verbose=1)

    # 训练数据记录
    total_losses = []
    timesteps = []
    total_rewards_list = []
    all_actions = []
    all_losses = []
    all_actual_storage_power_values = []
    all_total_charge_energy = []
    all_total_discharge_energy = []
    all_energy_imbalance = []

    # 训练循环
    for i in range(3):
        model.learn(total_timesteps=24)
        obs = env.reset()
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)

        all_actions.append(action.copy())
        all_losses.append(info["total_loss"])
        all_actual_storage_power_values.append(info["actual_storage_power_values"])

        total_losses.append(info["total_loss"])
        total_rewards_list.append(info["total_reward_and_penalty"])
        all_total_charge_energy.append(info["total_charge_energy"])
        all_total_discharge_energy.append(info["total_discharge_energy"])
        all_energy_imbalance.append(info["energy_imbalance"])
        timesteps.append(i + 1)

        if (i + 1) % 1 == 0:
            print(f"Training day {i + 1}, total line loss: {info['total_loss']:.4f}, "
                  f"reward+penalty: {info['total_reward_and_penalty']:.6f}, "
                  f"reward: {info['reward']:.4f}, penalty: {info['penalty']:.4f}")

    # 找到最优结果
    min_loss_idx = np.argmin(all_losses)
    best_action = all_actions[min_loss_idx]
    min_loss_value = all_losses[min_loss_idx]
    best_storage_power_profile = np.array(all_actual_storage_power_values[min_loss_idx])
    best_total_charge_energy = all_total_charge_energy[min_loss_idx]
    best_total_discharge_energy = all_total_discharge_energy[min_loss_idx]
    best_energy_imbalance = all_energy_imbalance[min_loss_idx]

    print(f"\n最优结果出现在第 {min_loss_idx + 1} 个训练批次")
    print(f"最低全天损耗: {min_loss_value:.4f} kW")
    print(f"全天充电量: {best_total_charge_energy:.4f} kWh")
    print(f"全天放电量: {best_total_discharge_energy:.4f} kWh")
    print(f"充放功率偏差: {best_energy_imbalance:.4f} kWh")

    # 创建所有图表
    create_all_plots(env, DEFAULT_PV_CONFIG, best_storage_power_profile,
                    best_total_charge_energy, best_total_discharge_energy, best_energy_imbalance,
                    total_losses, timesteps, total_rewards_list)

    # 最终测试
    obs = env.reset()
    test_action, _ = model.predict(obs)
    obs, reward, done, info = env.step(test_action)
    print(f"\n最终测试总损耗: {info['total_loss']:.4f} kW")

if __name__ == "__main__":
    main()
