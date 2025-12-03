# algorithms/epra_t.py

"""
EPRA-T：任务级价格的 Stackelberg 定价方案
"""

from typing import Dict, List, Tuple
from models.user_device import UserDevice
from models.edge_server import EdgeServer
from models.cloud_server import CloudServer
from models.task import Task
from algorithms.uta_g import uta_g_single_user, era_allocation
from utils.cost_functions import cost_local, cost_edge


def epra_t(
    user: UserDevice,
    edge: EdgeServer,
    cloud: CloudServer,
    cpl_candidates: List[float],
    mu_min: float,
    mu_max: float
) -> Tuple[Dict[Tuple[int, int], float], float]:
    """
    输入：一组 CPL 目标值，用来反推出不同任务的价格 μ_ij
    输出：(每个任务的最优价格字典 mu_edge, 最大收益 best_revenue)
    """
    best_mu_edge: Dict[Tuple[int, int], float] = {}
    best_revenue = 0.0

    for cpl_target in cpl_candidates:
        mu_edge: Dict[Tuple[int, int], float] = {}

        # 通过 CPL 目标值近似反解每个任务的 μ_ij
        for t in user.tasks:
            key = (t.user_id, t.task_id)
            U_loc = cost_local(t, user)
            U_edge0 = cost_edge(t, user, edge, 0.0)  # 价格为 0 时的边缘成本

            # 目标： CPL = (U_sub - U_loc)/Rij ~= cpl_target
            # 简化为： U_edge ≈ U_loc + cpl_target * Rij
            U_target = U_loc + cpl_target * t.Rij

            if t.wM * t.Cij > 0:
                mu_ij = (U_target - U_edge0) / (t.wM * t.Cij)
            else:
                mu_ij = mu_min

            mu_ij = max(mu_min, min(mu_max, mu_ij))
            mu_edge[key] = mu_ij

        # 用户决策 + ERA 资源分配
        decisions = uta_g_single_user(user, [edge], cloud, mu_edge)
        edge_tasks = [
            t for t in user.tasks
            if decisions[(t.user_id, t.task_id)].startswith("edge")
        ]
        revenue = era_allocation(edge, user, edge_tasks, mu_edge)

        if revenue > best_revenue:
            best_revenue = revenue
            best_mu_edge = mu_edge.copy()

    return best_mu_edge, best_revenue
