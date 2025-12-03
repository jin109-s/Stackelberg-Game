# algorithms/baseline.py

"""
Baseline 对比算法：
- Local-only   : 全部任务在本地执行（边缘收益为 0）
- Cloud-only   : 全部任务在云端执行（边缘收益为 0）
- Min-Cost     : 每个任务选择成本最低的位置
- Random       : 每个任务随机选择执行地
"""

import random
from 03_models.01_user_device import UserDevice
from 03_models.02_edge_server import EdgeServer
from 03_models.03_cloud_server import CloudServer
from 03_models.00_task import Task
from 04_utils.00_cost_functions import cost_local, cost_edge, cost_cloud


def baseline_local(user: UserDevice, edge: EdgeServer, cloud: CloudServer) -> float:
    """
    全部本地执行：边缘服务器不参与，边缘收益为 0
    """
    return 0.0


def baseline_cloud(user: UserDevice, edge: EdgeServer, cloud: CloudServer) -> float:
    """
    全部云执行：边缘服务器不参与，边缘收益为 0
    """
    return 0.0


def baseline_min_cost(user: UserDevice, edge: EdgeServer, cloud: CloudServer) -> float:
    """
    最小成本策略：
    每个任务比较 Local / Edge / Cloud 的用户代价，
    选择最便宜的一种。边缘收益只来自被分配到 Edge 的任务。
    """
    decisions = {}
    for t in user.tasks:
        Ul = cost_local(t, user)
        Ue = cost_edge(t, user, edge, 0.0)  # 假设价格给一个很小的值
        Uc = cost_cloud(t, cloud)
        place = min([(Ul, "local"), (Ue, "edge"), (Uc, "cloud")])[1]
        decisions[(t.user_id, t.task_id)] = place

    edge_tasks = [t for t in user.tasks if decisions[(t.user_id, t.task_id)] == "edge"]
    # 假设 baseline 的边缘价格为一个固定小值
    mu_baseline = 1e-11
    revenue = sum(mu_baseline * t.Cij for t in edge_tasks)
    return revenue


def baseline_random(user: UserDevice, edge: EdgeServer, cloud: CloudServer) -> float:
    """
    随机策略：
    每个任务随机选择 local / edge / cloud 之一。
    边缘收益来自随机分到 edge 的任务。
    """
    decisions = {}
    choices = ["local", "edge", "cloud"]
    for t in user.tasks:
        decisions[(t.user_id, t.task_id)] = random.choice(choices)

    edge_tasks = [t for t in user.tasks if decisions[(t.user_id, t.task_id)] == "edge"]
    mu_baseline = 1e-11
    revenue = sum(mu_baseline * t.Cij for t in edge_tasks)
    return revenue

