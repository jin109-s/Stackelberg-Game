# models/cloud_server.py

class CloudServer:
    """
    云服务器：
    - freq: CPU 频率
    - mu_c: 云的单价(每个CPU周期费用)
    """

    def __init__(self, freq, mu_c):
        self.freq = freq
        self.mu_c = mu_c
