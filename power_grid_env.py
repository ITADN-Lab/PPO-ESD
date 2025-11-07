import gym
from gym import spaces
import numpy as np
from config import bus_lists, Branch, Vbase, Sbase, Zbase, DEFAULT_PV_CONFIG
from pv_generation import generate_multi_pv_output
from power_flow import backward_forward_sweep, calculate_losses


class PowerGridEnv(gym.Env):
    def __init__(self, storage_node=18, pv_nodes_config=None, discharge_limit=1000, charge_limit=1000):
        super().__init__()
        self.num_nodes = len(bus_lists[0])
        self.Vbase = Vbase
        self.Sbase = Sbase
        self.Zbase = Zbase

        # 添加对 bus_lists 的引用
        self.bus_lists = bus_lists

        self.storage_capacity = 24000 / 1000.0 / Sbase
        self.storage_max_power = 1300 / 1000.0 / Sbase

        if pv_nodes_config is None:
            pv_nodes_config = DEFAULT_PV_CONFIG

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

        V, I = backward_forward_sweep(bus_data, branch_data)
        line_losses, total_loss = calculate_losses(branch_data, I)
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

        V, I = backward_forward_sweep(bus_data, branch_data)
        line_losses, total_loss = calculate_losses(branch_data, I)
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

            if power_discharge < 0:  # 充电
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
            elif power_discharge > 0:  # 放电
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
            "total_charge_energy": total_charge_energy * self.Sbase * 1000,
            "total_discharge_energy": total_discharge_energy * self.Sbase * 1000,
            "energy_imbalance": abs(total_charge_energy - total_discharge_energy) * self.Sbase * 1000
        }
        return obs, rewards + penalty, done, info