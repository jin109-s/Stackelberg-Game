# utils/cost_functions.py

"""
成本计算工具模块：
包含本地、边缘、云三种执行方式的成本公式
"""

import math
from 04_utils.02_network_params import B_UP_EDGE, B_DOWN_EDGE, B_UP_CLOUD, B_DOWN_CLOUD, P_TX, P_RX


# ========== 基础通信时间 ==========
def uplink_time(data_kb, bandwidth):
    return data_kb / bandwidth

def downlink_time(data_kb, bandwidth):
    return data_kb / bandwidth

# ========== 通信能耗 ==========
def uplink_energy(t):
    return P_TX * t

def downlink_energy(t):
    return P_RX * t


# ========== 成本函数：本地执行 ==========
def cost_local(task, user):
    """
    用户本地计算成本：
    时间 + 能耗
    """
    T_comp = task.Cij / user.freq
    E_comp = user.beta * task.Cij * (user.freq ** 2)
    return task.wT * T_comp + task.wE * E_comp


# ========== 成本函数：边缘执行 ==========
def cost_edge(task, user, edge, mu):
    """
    边缘执行成本包含：
    - 上下行通信时间
    - 通信能耗
    - 边缘计算时间
    - 金钱成本（价格 × CPU周期）
    """
    t_up = uplink_time(task.Din, B_UP_EDGE)
    t_down = downlink_time(task.Dout, B_DOWN_EDGE)

    T_comm = t_up + t_down
    E_comm = uplink_energy(t_up) + downlink_energy(t_down)
    T_comp = task.Cij / edge.freq

    return (
        task.wT * (T_comm + T_comp)
        + task.wE * E_comm
        + task.wM * (mu * task.Cij)
    )


# ========== 成本函数：云执行 ==========
def cost_cloud(task, cloud):
    """
    云执行成本：
    - 上下行通信
    - 请求云计算
    - 云端单价 × CPU周期
    """
    t_up = uplink_time(task.Din, B_UP_CLOUD)
    t_down = downlink_time(task.Dout, B_DOWN_CLOUD)

    T_comm = t_up + t_down
    E_comm = uplink_energy(t_up) + downlink_energy(t_down)
    T_comp = task.Cij / cloud.freq

    return (
        task.wT * (T_comm + T_comp)
        + task.wE * E_comm
        + task.wM * (cloud.mu_c * task.Cij)
    )

