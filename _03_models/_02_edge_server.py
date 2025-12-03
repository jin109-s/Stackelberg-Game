# models/edge_server.py

class EdgeServer:
    """
    边缘服务器：
    - freq: CPU 频率
    - total_resource: 总资源
    - per_user_resource_max: 给每个用户分配的最大资源
    """

    def __init__(self, server_id, freq, total_resource):
        self.server_id = server_id
        self.freq = freq
        self.total_resource = total_resource
        self.per_user_resource_max = {}
