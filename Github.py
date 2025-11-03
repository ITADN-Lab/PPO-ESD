import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from matplotlib import rcParams
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 18,
    'axes.labelsize': 20,
    'axes.titlesize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 20,
    'legend.fontsize': 15,
    'figure.dpi': 300,
    'figure.figsize': (8, 4.5),
    'savefig.format': 'png',
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.2,
    'grid.linestyle': '--'
})

SAVE_DIR = r"E:\UJN\代码\结果图\SCI33节点多光伏充放平衡"
os.makedirs(SAVE_DIR, exist_ok=True)

Bus = np.array([
    [1, 0, 0], [2, 100, 60], [3, 90, 40], [4, 120, 80], [5, 60, 30],
    [6, 60, 20], [7, 200, 100], [8, 200, 100], [9, 60, 20], [10, 60, 20],
    [11, 45, 30], [12, 60, 35], [13, 60, 35], [14, 120, 80], [15, 60, 10],
    [16, 60, 20], [17, 60, 20], [18, 90, 40], [19, 90, 40], [20, 90, 40],
    [21, 90, 40], [22, 90, 40], [23, 90, 40], [24, 420, 200], [25, 420, 200],
    [26, 60, 25], [27, 60, 25], [28, 60, 20], [29, 120, 70], [30, 200, 600],
    [31, 150, 70], [32, 210, 100], [33, 60, 40]
], dtype=float)

Branch = np.array([
    [1, 1, 2, 0.0922, 0.0407], [2, 2, 3, 0.4930, 0.2511], [3, 3, 4, 0.3660, 0.1864],
    [4, 4, 5, 0.3811, 0.1941], [5, 5, 6, 0.8190, 0.7070], [6, 6, 7, 0.1872, 0.6188],
    [7, 7, 8, 0.7144, 0.2351], [8, 8, 9, 1.0300, 0.7400], [9, 9, 10, 1.0440, 0.7400],
    [10, 10, 11, 0.1966, 0.065], [11, 11, 12, 0.3744, 0.1238], [12, 12, 13, 1.4680, 1.1550],
    [13, 13, 14, 0.5416, 0.7129], [14, 14, 15, 0.5910, 0.5260], [15, 15, 16, 0.7463, 0.5450],
    [16, 16, 17, 1.2890, 1.7210], [17, 17, 18, 0.7320, 0.5740], [18, 2, 19, 0.1640, 0.1565],
    [19, 19, 20, 1.5042, 1.3554], [20, 20, 21, 0.4095, 0.4784], [21, 21, 22, 0.7089, 0.9373],
    [22, 3, 23, 0.4512, 0.3083], [23, 23, 24, 0.8980, 0.7091], [24, 24, 25, 0.8960, 0.7011],
    [25, 6, 26, 0.2030, 0.1034], [26, 26, 27, 0.2842, 0.1447], [27, 27, 28, 1.0590, 0.9337],
    [28, 28, 29, 0.8042, 0.7006], [29, 29, 30, 0.5075, 0.2585], [30, 30, 31, 0.9744, 0.9630],
    [31, 31, 32, 0.3105, 0.3619], [32, 32, 33, 0.3410, 0.5302]
])

Vbase = 12.66  # kV
Sbase = 100.0  # MVA
Zbase = Vbase ** 2 / Sbase

coefficients = 1.0 * np.array([0.25, 0.2, 0.15, 0.1, 0.05, 0.05, 0.2, 0.3, 0.4, 0.55,
                               0.6, 0.7, 0.75, 0.7, 0.65, 0.62, 0.75, 0.8, 0.9, 0.95,
                               1.0, 0.8, 0.7, 0.4])

bus_lists = []
for coeff in coefficients:
    new_bus = Bus.copy()
    new_bus[:, 1] *= coeff
    new_bus[:, 2] *= coeff
    bus_lists.append(new_bus)


def generate_multi_pv_output(pv_nodes_config):

    base_pv_curve = np.array([0, 0, 0, 0, 50, 80, 90, 100, 150, 400, 500, 600,
                              700, 700, 650, 500, 400, 100, 50, 0, 0, 0, 0, 0])

    pv_outputs = {}
    for node, capacity_ratio in pv_nodes_config.items():

        node_pv_output = capacity_ratio * base_pv_curve
        pv_outputs[node] = node_pv_output / 1000.0 / Sbase

    return pv_outputs


class PowerGridEnv(gym.Env):
    def __init__(self, storage_node=18, pv_nodes_config=None, discharge_limit=1000, charge_limit=1000):
        super().__init__()
        self.num_nodes = len(Bus)
        self.Vbase = Vbase
        self.Sbase = Sbase
        self.Zbase = Zbase

        self.storage_capacity = 24000 / 1000.0 / Sbase
        self.storage_max_power = 1300 / 1000.0 / Sbase

        if pv_nodes_config is None:

            pv_nodes_config = {
                18: 2.00,
                25: 2.00,
                12: 2.00,
                30: 2.00
            }

        self.pv_nodes_config = pv_nodes_config
        self.pv_outputs = generate_multi_pv_output(pv_nodes_config)
        self.storage_node = storage_node
        self.discharge_limit = discharge_limit
        self.charge_limit = charge_limit

        self.voltage_limits = [0.95, 1.05]

        self.observation_space = spaces.Box(low=0.0, high=1.5, shape=(self.num_nodes * 5 + 1,), dtype=np.float32)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(24,), dtype=np.float32)

        self.storage_power_values = []

        self.reset()

    def reset(self):
        self.current_soc = 0.5
        self.discharge_count = 0
        self.charge_count = 0
        self.line_losses_over_day = []
        self.previous_total_loss = None
        self.storage_power_values = []

        return self._get_observation(0, self.current_soc)

    def _get_observation(self, hour, soc):
        voltage = np.ones(self.num_nodes)
        current_bus = bus_lists[hour]
        demand = current_bus[:, 1] / (1000.0 * self.Sbase)

        pv_power = np.zeros(self.num_nodes)
        for node, pv_24h in self.pv_outputs.items():
            pv_power[node - 1] = pv_24h[hour]

        discharge_counts_obs = np.full(self.num_nodes, self.discharge_count)
        charge_counts_obs = np.full(self.num_nodes, self.charge_count)

        observation = np.concatenate((
            voltage,
            demand,
            pv_power,
            discharge_counts_obs,
            charge_counts_obs,
            [soc]
        ))

        return observation.astype(np.float32)

    def backward_forward_sweep(self, bus_data, branch_data, max_iter=100, tol=1e-6):
        nodes = sorted(bus_data.keys())
        V = {bus: complex(1.0, 0.0) for bus in nodes}
        I = {i: complex(0.0, 0.0) for i in range(len(branch_data))}

        for _ in range(max_iter):
            V_prev = V.copy()

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

            for i in range(len(branch_data)):
                line = branch_data[i]
                from_bus = line['from']
                to_bus = line['to']
                V[to_bus] = V[from_bus] - I[i] * line['Z_pu']

            max_diff = max(abs(V[bus] - V_prev[bus]) for bus in nodes)
            if max_diff < tol:
                break

        return V, I

    def calculate_losses(self, branch_data, I):
        line_losses = []
        total_loss = 0.0
        for idx, line in enumerate(branch_data):
            I_mag = abs(I[idx])
            loss_pu = (I_mag ** 2) * line['R_pu']
            loss_MW = loss_pu * Sbase
            line_losses.append(loss_MW)
            total_loss += loss_MW
        return line_losses, total_loss

    def calculate_power_flow(self, hour, storage_power):
        current_bus = bus_lists[hour]
        bus_data = {}
        for row in current_bus:
            bus = int(row[0])
            P_kW = row[1]
            Q_kVAR = row[2]
            bus_data[bus] = {
                'P_pu': P_kW / (1000.0 * self.Sbase),
                'Q_pu': Q_kVAR / (1000.0 * self.Sbase)
            }

        for node, pv_24h in self.pv_outputs.items():
            bus_data[node]['P_pu'] -= pv_24h[hour]

        bus_data[self.storage_node]['P_pu'] -= storage_power

        branch_data = []
        for row in Branch:
            branch_data.append({
                'from': int(row[1]),
                'to': int(row[2]),
                'R_pu': row[3] / self.Zbase,
                'X_pu': row[4] / self.Zbase,
                'Z_pu': complex(row[3] / self.Zbase, row[4] / self.Zbase)
            })

        V, I = self.backward_forward_sweep(bus_data, branch_data)
        line_losses, total_loss = self.calculate_losses(branch_data, I)
        voltages = np.array([abs(V[bus]) for bus in sorted(V.keys())])

        return voltages, line_losses, total_loss

    def calculate_losses_without_storage_with_pv(self, hour):

        current_bus = bus_lists[hour]
        bus_data = {}

        for row in current_bus:
            bus = int(row[0])
            P_kW = row[1]
            Q_kVAR = row[2]
            bus_data[bus] = {
                'P_pu': P_kW / (1000.0 * self.Sbase),
                'Q_pu': Q_kVAR / (1000.0 * self.Sbase)
            }

        for node, pv_24h in self.pv_outputs.items():
            bus_data[node]['P_pu'] -= pv_24h[hour]

        branch_data = []
        for row in Branch:
            branch_data.append({
                'from': int(row[1]),
                'to': int(row[2]),
                'R_pu': row[3] / self.Zbase,
                'X_pu': row[4] / self.Zbase,
                'Z_pu': complex(row[3] / self.Zbase, row[4] / self.Zbase)
            })

        V, I = self.backward_forward_sweep(bus_data, branch_data)
        line_losses, total_loss = self.calculate_losses(branch_data, I)

        return total_loss

    def step(self, action):

        normalized_action = action - np.mean(action)

        normalized_action = np.clip(normalized_action, -1.0, 1.0)

        soc = 0.5
        charge_cnt = 0
        discharge_cnt = 0
        total_charge_energy = 0.0
        total_discharge_energy = 0.0

        total_loss = 0.0
        voltages_day = []
        rewards = 0.0
        penalty = 0.0

        actual_storage_power_values = []

        for hour in range(24):

            power_discharge = normalized_action[hour] * self.storage_max_power
            actual_storage_power = 0.0

            if power_discharge < 0:
                if charge_cnt >= self.charge_limit:
                    penalty -= 50
                    actual_storage_power = 0.0
                else:
                    max_charge_possible = min(abs(power_discharge), (0.8 - soc) * self.storage_capacity)
                    if max_charge_possible > 0:
                        soc += max_charge_possible / self.storage_capacity
                        actual_storage_power = -max_charge_possible
                        charge_cnt += 1
                        total_charge_energy += max_charge_possible
                    else:
                        penalty -= 50
                        actual_storage_power = 0.0
            elif power_discharge > 0:
                if discharge_cnt >= self.discharge_limit:
                    penalty -= 50
                    actual_storage_power = 0.0
                else:
                    max_discharge_possible = min(power_discharge, (soc - 0.2) * self.storage_capacity)
                    if max_discharge_possible > 0:
                        soc -= max_discharge_possible / self.storage_capacity
                        actual_storage_power = max_discharge_possible
                        discharge_cnt += 1
                        total_discharge_energy += max_discharge_possible
                    else:
                        penalty -= 50
                        actual_storage_power = 0.0
            else:
                actual_storage_power = 0.0

            soc = np.clip(soc, 0.2, 0.8)
            voltages, line_losses, loss = self.calculate_power_flow(hour, actual_storage_power)
            voltages_day.append(voltages)
            total_loss += loss
            rewards += -loss * 1e4
            storage_power_kw = actual_storage_power * self.Sbase * 1000
            actual_storage_power_values.append(storage_power_kw)

        if self.previous_total_loss is not None:
            diff = total_loss * 1e3 - self.previous_total_loss * 1e3
            factor = 3e1 * abs(diff)
            if diff > 0:
                rewards -= factor
            else:
                rewards += factor

        self.previous_total_loss = total_loss

        self.current_soc = soc
        self.discharge_count = discharge_cnt
        self.charge_count = charge_cnt
        self.line_losses_over_day.append(total_loss)

        done = True

        obs = self._get_observation(23, self.current_soc)
        obs[:self.num_nodes] = voltages_day[-1]

        info = {
            "total_loss": total_loss * 1e3,
            "current_soc": self.current_soc,
            "discharge_count": self.discharge_count,
            "charge_count": self.charge_count,
            "reward": rewards,
            "penalty": penalty,
            "total_reward_and_penalty": rewards + penalty,
            "actual_storage_power_values": actual_storage_power_values,
            "total_charge_energy": total_charge_energy * self.Sbase * 1000,  # kW·h
            "total_discharge_energy": total_discharge_energy * self.Sbase * 1000,  # kW·h
            "energy_imbalance": abs(total_charge_energy - total_discharge_energy) * self.Sbase * 1000  # kW·h
        }
        return obs, rewards + penalty, done, info


if __name__ == "__main__":

    multi_pv_config = {
        18: 2.00,
        25: 2.00,
        12: 2.00,
        30: 2.00
    }

    env = PowerGridEnv(storage_node=18, pv_nodes_config=multi_pv_config,
                       discharge_limit=1000, charge_limit=1000)

    model = PPO("MlpPolicy", env,
                learning_rate=3e-3,
                n_steps=96,
                batch_size=48,
                verbose=1)

    total_losses = []
    timesteps = []
    total_rewards_list = []

    all_actions = []
    all_losses = []
    all_actual_storage_power_values = []
    all_total_charge_energy = []
    all_total_discharge_energy = []
    all_energy_imbalance = []

    for i in range(1000):
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
            print(
                f"Training day {i + 1}, total line loss: {info['total_loss']:.4f}, "
                f"reward+penalty: {info['total_reward_and_penalty']:.6f}, reward: {info['reward']:.4f}, penalty: {info['penalty']:.4f}")

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

    colors = plt.cm.tab10.colors

    hours = range(24)

    plt.figure()
    for node, ratio in multi_pv_config.items():
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
    plt.savefig(os.path.join(SAVE_DIR, "6IEEE33多节点光伏_光伏出力曲线.png"), dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure()
    charge_power = np.where(best_storage_power_profile < 0, best_storage_power_profile, 0)
    discharge_power = np.where(best_storage_power_profile > 0, best_storage_power_profile, 0)

    colors = plt.cm.tab10.colors
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
    plt.savefig(os.path.join(SAVE_DIR, "6IEEE33多节点光伏_最优储能调度策略.png"), dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n最优储能功率调度 (kW):")
    for hour in range(24):
        power = best_storage_power_profile[hour]
        if power > 0:
            print(f"第{hour:2d}小时: 放电 {power:6.2f} kW")
        elif power < 0:
            print(f"第{hour:2d}小时: 充电 {abs(power):6.2f} kW")
        else:
            print(f"第{hour:2d}小时: 无动作  0.00 kW")

    losses_without_storage_with_pv = []
    best_losses_with_storage = []
    demand_without_storage = []
    demand_with_storage = []

    for hour in range(24):

        total_loss_no_storage = env.calculate_losses_without_storage_with_pv(hour)
        losses_without_storage_with_pv.append(total_loss_no_storage * 1e3)


        current_bus = bus_lists[hour]
        demand_no_storage = sum(current_bus[:, 1])
        demand_without_storage.append(demand_no_storage)

        storage_power = best_storage_power_profile[hour] / (env.Sbase * 1000)
        total_loss_with_storage = env.calculate_power_flow(hour, storage_power)[2]
        best_losses_with_storage.append(total_loss_with_storage * 1e3)

        demand_with_storage.append(demand_no_storage)

    loss_ratios_without_storage = [loss / demand * 100 if demand > 0 else 0 for loss, demand in
                                   zip(losses_without_storage_with_pv, demand_without_storage)]
    loss_ratios_with_storage = [loss / demand * 100 if demand > 0 else 0 for loss, demand in
                                zip(best_losses_with_storage, demand_with_storage)]

    loss_data_path = os.path.join(SAVE_DIR, "6负载系数1.0,光伏系数2.00,IEEE33多节点光伏_最优储能调度线损数据.txt")
    with open(loss_data_path, 'w', encoding='utf-8') as f:
        f.write("小时\t线损(kW)\n")
        for hour, loss in enumerate(best_losses_with_storage):
            f.write(f"{hour}\t{loss:.4f}\n")
    print(f"最优储能调度线损数据已保存至: {loss_data_path}")

    loss_ratio_path = os.path.join(SAVE_DIR, "6负载系数1.0,光伏系数2.00,IEEE33多节点光伏_最优储能调度线损率数据.txt")
    with open(loss_ratio_path, 'w', encoding='utf-8') as f:
        f.write("小时\t线损率(%)\n")
        for hour, loss_ratio in enumerate(loss_ratios_with_storage):
            f.write(f"{hour}\t{loss_ratio:.4f}\n")
    print(f"最优储能调度线损率数据已保存至: {loss_ratio_path}")

    loss_data_path_without = os.path.join(SAVE_DIR, "6未接储能负载系数1.0,光伏系数2.00,IEEE33多节点光伏_最优储能调度线损数据.txt")
    with open(loss_data_path_without, 'w', encoding='utf-8') as f:
        f.write("小时\t线损(kW)\n")
        for hour, loss in enumerate(losses_without_storage_with_pv):
            f.write(f"{hour}\t{loss:.4f}\n")
    print(f"未接储能线损数据已保存至: {loss_data_path_without}")

    loss_ratio_path_without = os.path.join(SAVE_DIR, "6未接储能负载系数1.0,光伏系数2.00,IEEE33多节点光伏_最优储能调度线损率数据.txt")
    with open(loss_ratio_path_without, 'w', encoding='utf-8') as f:
        f.write("小时\t线损率(%)\n")
        for hour, loss_ratio in enumerate(loss_ratios_without_storage):
            f.write(f"{hour}\t{loss_ratio:.4f}\n")
    print(f"未接储能线损率数据已保存至: {loss_ratio_path_without}")

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
    plt.savefig(os.path.join(SAVE_DIR, "6IEEE33多节点光伏_总线损耗对比图.png"), dpi=300, bbox_inches='tight')
    plt.show()

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
    plt.savefig(os.path.join(SAVE_DIR, "6IEEE33多节点光伏_线损率对比图.png"), dpi=300, bbox_inches='tight')
    plt.show()

    all_total_loss_no_storage = sum(losses_without_storage_with_pv)
    all_total_loss_with_storage = sum(best_losses_with_storage)
    loss_reduction = all_total_loss_no_storage - all_total_loss_with_storage
    loss_reduction_percent = (loss_reduction / all_total_loss_no_storage) * 100

    total_demand_no_storage = sum(demand_without_storage)
    total_demand_with_storage = sum(demand_with_storage)
    overall_loss_ratio_no_storage = (all_total_loss_no_storage / total_demand_no_storage) * 100
    overall_loss_ratio_with_storage = (all_total_loss_with_storage / total_demand_with_storage) * 100

    print(f"\n多节点光伏系统损耗统计:")
    print(f"未接入储能全天损耗: {all_total_loss_no_storage:.2f} kW")
    print(f"光储协同全天损耗: {all_total_loss_with_storage:.2f} kW")
    print(f"损耗降低: {loss_reduction:.2f} kW ({loss_reduction_percent:.2f}%)")
    print(f"未接储能全天线损率: {overall_loss_ratio_no_storage:.2f}%")
    print(f"光储协同全天线损率: {overall_loss_ratio_with_storage:.2f}%)")

    obs = env.reset()
    test_action, _ = model.predict(obs)
    obs, reward, done, info = env.step(test_action)
    print(f"\n最终测试总损耗: {info['total_loss']:.4f} kW")

    plt.figure(figsize=(8, 4.5))
    plt.plot(timesteps, total_losses, color=colors[0], label='Line Loss')
    plt.xlabel('Training Days', fontweight='bold')
    plt.ylabel('Line Loss (kW)', fontweight='bold')
    plt.title('Total Line Loss(Multi-Node PV Access)', fontweight='bold')
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.legend(frameon=True, fancybox=False, edgecolor='black')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "6IEEE33多节点光伏训练线损曲线total_losses_24h_action.png"))
    plt.show()

    plt.figure(figsize=(8, 4.5))
    plt.plot(timesteps, total_rewards_list, color=colors[1], label='Total Reward')
    plt.xlabel('Training Days', fontweight='bold')
    plt.ylabel('Reward Value', fontweight='bold')
    plt.title('Total Reward(Multi-Node PV Access)', fontweight='bold')
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.legend(frameon=True, fancybox=False, edgecolor='black')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "6IEEE33多节点光伏奖励曲线reward_and_penalty_curve.png"))
    plt.show()

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
    fig.savefig(os.path.join(SAVE_DIR, "6IEEE33多节点光伏线损与奖励曲线总图.png"))
    plt.show()

    summary_data = [
        ["Total Loss without ESS", f"{all_total_loss_no_storage:.2f} kW"],
        ["Total Loss with ESS", f"{all_total_loss_with_storage:.2f} kW"],
        ["Loss Reduction", f"{loss_reduction:.2f} kW ({loss_reduction_percent:.2f}%)"],
        ["Overall Loss Rate without ESS", f"{overall_loss_ratio_no_storage:.2f}%"],
        ["Overall Loss Rate with ESS", f"{overall_loss_ratio_with_storage:.2f}%"],
        ["Total Charging Energy", f"{best_total_charge_energy:.2f} kWh"],
        ["Total Discharging Energy", f"{best_total_discharge_energy:.2f} kWh"],
        ["Energy Imbalance", f"{best_energy_imbalance:.2f} kWh"]
    ]

    with open(os.path.join(SAVE_DIR, "6IEEE33多节点结果数据记录.txt"), "w") as f:
        f.write("Performance Summary\n")
        f.write("=" * 40 + "\n")
        for row in summary_data:
            f.write(f"{row[0]:<35} {row[1]}\n")
