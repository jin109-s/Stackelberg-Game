# models/task.py

class Task:
    """
    任务类：
    - Cij: 所需 CPU 周期
    - Din: 输入数据大小(KB)
    - Dout: 输出数据大小(KB)
    - Rij: 占用资源（比如带宽/CPU比例）
    - wT, wE, wM: 时间/能耗/金钱 的权重
    """

    def __init__(self, user_id, task_id, Cij, Din, Dout, Rij, wT, wE, wM):
        self.user_id = user_id
        self.task_id = task_id
        self.Cij = Cij
        self.Din = Din
        self.Dout = Dout
        self.Rij = Rij
        self.wT = wT
        self.wE = wE
        self.wM = wM
