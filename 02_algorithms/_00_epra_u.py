# algorithms/epra_u.py

"""
EPRA-U：统一价格的 Stackelberg 定价方案
"""

from typing import List, Tuple
from 03_models.01_user_device import UserDevice
from 03_models.02_edge_server import EdgeServer
from 03_models.03_cloud_server import CloudServer
from 02_algorithms.02_uta_g import uta_g_single_user, era_allocation


def epra_u(
    user: UserDevice,
    edge: EdgeServer,
    cloud: CloudServer,
    mu_candidates: List[float]
) -> Tuple[float, float]:
    """
    输入：一组候选单价 mu 列表
    输出：(最优价格 best_mu, 最大收益 best_revenue)
    """
    best_mu = mu_candidates[0]
    best_revenue = 0.0

    for mu in mu_candidates:
        # 给每个任务同样的价格 mu
        mu_edge = {(t.user_id, t.task_id): mu for t in user.tasks}

        # 用户侧任务决策
        decisions = uta_g_single_user(user, [edge], cloud, mu_edge)

        # 挑出真正跑在边缘的任务
        edge_tasks = [
            t for t in user.tasks
            if decisions[(t.user_id, t.task_id)].startswith("edge")
        ]

        # ERA 分配资源 + 计算收益
        revenue = era_allocation(edge, user, edge_tasks, mu_edge)

        if revenue > best_revenue:
            best_revenue = revenue
            best_mu = mu

    return best_mu, best_revenue

