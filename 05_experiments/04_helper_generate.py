# experiments/helper_generate.py

"""
用户、任务、边缘与云服务器的生成函数
"""

import random
from models.user_device import UserDevice
from models.task import Task
from models.edge_server import EdgeServer
from models.cloud_server import CloudServer


def generate_user(user_id, num_tasks, local_factor=1.0, input_scale=1.0):
    freq = 1.5e9
    beta = 1e-27
    base_res = 150.0

    user = UserDevice(user_id, freq, beta, base_res * local_factor)

    for tid in range(num_tasks):
        C = random.uniform(1e8, 5e8)
        Din = random.uniform(500, 2000) * input_scale
        Dout = random.uniform(100, 500) * input_scale
        Rij = random.uniform(5, 20)
        wM = random.uniform(0.2, 0.8)
        remain = 1 - wM
        wT = random.uniform(remain * 0.4, remain * 0.6)
        wE = remain - wT

        task = Task(user_id, tid, C, Din, Dout, Rij, wT, wE, wM)
        user.add_task(task)

    return user


def build_edge_cloud(edge_factor=1.0):
    base_total = 500.0
    per_user = 200.0

    edge = EdgeServer(0, 4e9, base_total * edge_factor)
    edge.per_user_resource_max[0] = per_user * edge_factor

    cloud = CloudServer(2.4e9, 6e-12)
    return edge, cloud

