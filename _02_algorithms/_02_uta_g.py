# algorithms/uta_g.py

"""
UTA-G 用户侧任务决策算法 + ERA 边缘资源分配
"""

from typing import Dict, List, Tuple
from _03_models._01_user_device import UserDevice
from _03_models._02_edge_server import EdgeServer
from _03_models._03_cloud_server import CloudServer
from _03_models._00_task import Task
from _04_utils._00_cost_functions import cost_local, cost_edge, cost_cloud


def uta_g_single_user(
    user: UserDevice,
    edges: List[EdgeServer],
    cloud: CloudServer,
    mu_edge: Dict[Tuple[int, int], float]
) -> Dict[Tuple[int, int], str]:
    """
    UTA-G 用户决策：
    返回 (user_id, task_id) -> 'local' / 'edge_x' / 'cloud'
    """
    decisions: Dict[Tuple[int, int], str] = {}
    edge = edges[0]  # 简化：单边缘服务器

    # 1) 不考虑资源限制，先看每个任务最便宜在哪跑
    Uvals = {}
    best_place = {}
    for t in user.tasks:
        key = (t.user_id, t.task_id)
        mu = mu_edge[key]
        Ul = cost_local(t, user)
        Ue = cost_edge(t, user, edge, mu)
        Uc = cost_cloud(t, cloud)
        Uvals[key] = (Ul, Ue, Uc)
        best_place[key] = min([(Ul, "local"), (Ue, "edge"), (Uc, "cloud")])[1]

    # 2) 设备侧迁移 —— 本地资源是否够
    local_tasks = [t for t in user.tasks if best_place[(t.user_id, t.task_id)] == "local"]
    total_local_R = sum(t.Rij for t in local_tasks)

    stay_local = set()
    migrate_from_local = set()

    if total_local_R <= user.local_resource_max:
        # 本地资源够用：全部留在本地
        for t in local_tasks:
            stay_local.add((t.user_id, t.task_id))
    else:
        # 本地资源不够：按 CPL 贪心选择保留在本地的任务
        CPL_list = []
        for t in local_tasks:
            key = (t.user_id, t.task_id)
            Ul, Ue, Uc = Uvals[key]
            U_sub = min(Ue, Uc)
            CPL_val = (U_sub - Ul) / t.Rij
            CPL_list.append((CPL_val, t))

        CPL_list.sort(key=lambda x: x[0], reverse=True)
        remaining = user.local_resource_max
        for cpl, t in CPL_list:
            key = (t.user_id, t.task_id)
            if t.Rij <= remaining and cpl > 0:
                stay_local.add(key)
                remaining -= t.Rij
            else:
                migrate_from_local.add(key)

    # 标记本地执行的任务
    for key in stay_local:
        decisions[key] = "local"

    # 从本地迁出的任务，在 edge 和 cloud 之间选更便宜的
    for key in migrate_from_local:
        uid, tid = key
        t = next(x for x in user.tasks if x.task_id == tid)
        mu = mu_edge[key]
        Ue = cost_edge(t, user, edge, mu)
        Uc = cost_cloud(t, cloud)
        decisions[key] = f"edge_{edge.server_id}" if Ue <= Uc else "cloud"

    # 3) 一开始就不是本地最优的任务：直接按 best_place
    for t in user.tasks:
        key = (t.user_id, t.task_id)
        if key in decisions:
            continue
        if best_place[key] == "edge":
            decisions[key] = f"edge_{edge.server_id}"
        else:
            decisions[key] = best_place[key]

    # 4) 边缘资源检查：ERA + CPE 贪心
    edge_tasks = [t for t in user.tasks if decisions[(t.user_id, t.task_id)].startswith("edge")]
    R_edge_max = edge.per_user_resource_max.get(user.user_id, edge.total_resource)
    total_edge_R = sum(t.Rij for t in edge_tasks)

    if total_edge_R <= R_edge_max:
        return decisions

    # 边缘资源不够，按 CPE 排序
    CPE_list = []
    for t in edge_tasks:
        key = (t.user_id, t.task_id)
        _, Ue, Uc = Uvals[key]
        CPE_val = (Uc - Ue) / t.Rij
        CPE_list.append((CPE_val, t))
    CPE_list.sort(key=lambda x: x[0], reverse=True)

    remaining_edge_R = R_edge_max
    stay_edge = set()
    migrate_to_cloud = set()

    for cpe, t in CPE_list:
        key = (t.user_id, t.task_id)
        if t.Rij <= remaining_edge_R and cpe > 0:
            stay_edge.add(key)
            remaining_edge_R -= t.Rij
        else:
            migrate_to_cloud.add(key)

    for key in migrate_to_cloud:
        decisions[key] = "cloud"

    return decisions


def era_allocation(
    edge: EdgeServer,
    user: UserDevice,
    task_set: List[Task],
    mu_edge: Dict[Tuple[int, int], float]
) -> float:
    """
    ERA 资源分配（MCP 贪心）：
    返回该用户在该边缘服务器上的总收益
    """
    # MCP = mu_ij * Cij / Rij
    mcp_list = []
    for t in task_set:
        key = (t.user_id, t.task_id)
        mu = mu_edge[key]
        mcp_list.append((mu * t.Cij / t.Rij, t))

    mcp_list.sort(key=lambda x: x[0], reverse=True)

    remaining = edge.per_user_resource_max.get(user.user_id, edge.total_resource)
    revenue = 0.0

    for _, t in mcp_list:
        key = (t.user_id, t.task_id)
        if t.Rij <= remaining:
            remaining -= t.Rij
            revenue += mu_edge[key] * t.Cij

    return revenue


