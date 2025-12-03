# Stackelberg-Game
## 这是一个按照某个论文内容的编写的代码，以下是整个代码目录:
EPRA-Stackelberg-Replication/  
│  
├── models/  
│   ├── **task.py（定义任务的数据模型）**  
│   ├── **user_device.py（定义用户设备的数据模型）**  
│   ├── **edge_server.py（定义边缘服务器的数据类型）**  
│   └── **cloud_server.py（定义云服务器的数据模型）**  
│  
├── algorithms/  
│   ├── **epra_u.py（实现EPRA-U算法）**  
│   ├── **epra_t.py（实现EPRA-T算法）**  
│   ├── **uta_g.py（实现UTA-G算法）**  
│   └── **baseline.py（实现四种基线算法）**  
│  
├── utils/  
│   ├── **cost_functions.py（实现各类成本计算函数）**  
│   └── **plot_utils.py（提供绘图工具，用于生成实验结果的折线图）**  
│   └── **network_params.py（定义系统的网络参数和物理常数）**  
│  
├── experiments/  
│   ├── **exp1_tasks.py（实现任务数量对边缘收益的影响实验）**  
│   ├── **exp2_local_resource.py（实现本地资源系数对边缘收益的影响实验）**  
│   ├── **exp3_edge_resource.py（实现边缘系统对边缘收益的影响实验）**  
│   └── **exp4_input_scale.py（实现输入数据大小对边缘收益的影响实验）**  
│   └── **helper_generate.py（提供实验所需的辅助生成函数）**  
│  
├── **main.py（程序入口文件，用于调用并执行所有实验）**  
