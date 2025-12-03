# models/user_device.py

class UserDevice:
    """
    用户设备类：
    - freq: 本地 CPU 频率
    - beta: 电容系数（能耗参数）
    - local_resource_max: 本地可用资源上限
    - tasks: 用户的所有任务
    """

    def __init__(self, user_id, freq, beta, local_resource_max):
        self.user_id = user_id
        self.freq = freq
        self.beta = beta
        self.local_resource_max = local_resource_max
        self.tasks = []

    def add_task(self, task):
        self.tasks.append(task)
