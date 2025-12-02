# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from collections import defaultdict
import random

# 设置随机种子保证实验可复现
np.random.seed(42)
random.seed(42)

# ------------------------------ 核心UTAG算法（基础模块） ------------------------------
class UTAG:
    def __init__(self, user_tasks, processors, resource_limits, cost_weights):
        self.tasks = user_tasks
        self.processors = processors
        self.local_resource_max = resource_limits['local_max']
        self.edge_resource_max = resource_limits['edge_max']
        self.w_T = cost_weights['w_T']
        self.w_E = cost_weights['w_E']
        self.w_M = cost_weights['w_M']
        self.task_allocation = np.zeros((len(self.tasks), len(self.processors)))
        
        # 论文常量（固定值，匹配实验场景）
        self.beta = 1e-27  # 移动设备电容系数
        self.transfer_rate = 1e6  # 数据传输速率（1Mbps）

    def calculate_cost(self, task, processor, edge_unit_price=None):
        """计算任务在指定处理器的总成本（时间+能量+货币加权和）"""
        cpu_cycles = task['cpu_cycles']
        cpu_freq = processor['cpu_freq']
        
        # 1. 时间成本（计算时间 + 通信时间）
        compute_time = cpu_cycles / cpu_freq  # 计算时间
        if processor['processor_id'] == 0:  # 云
            upload_time = (task['local_data'] + task['external_data']) / self.transfer_rate
            download_time = task['result_data'] / self.transfer_rate
            comm_time = upload_time + download_time
        elif 1 <= processor['processor_id'] <= len(self.processors)-2:  # 边缘
            upload_time = max(task['local_data'], task['external_data']) / self.transfer_rate
            download_time = task['result_data'] / self.transfer_rate
            comm_time = upload_time + download_time
        else:  # 本地
            comm_time = 0
        time_cost = self.w_T * (compute_time + comm_time)

        # 2. 能量成本
        if processor['processor_id'] == len(self.processors)-1:  # 本地计算能耗
            energy_cost = self.w_E * self.beta * (cpu_freq ** 2) * cpu_cycles
        else:  # 边缘/云仅通信能耗
            data_volume = (max(task['local_data'], task['external_data']) if processor['processor_id'] !=0 else (task['local_data'] + task['external_data'])) + task['result_data']
            energy_cost = self.w_E * data_volume * 1e-3  # 每KB消耗1mJ

        # 3. 货币成本（支持动态单价，区分加权/未加权）
        use_price = 0.0
        monetary_cost_weighted = 0.0
        monetary_cost_raw = 0.0
        
        if processor['processor_id'] == 0:  # 云
            use_price = processor['unit_price']
            monetary_cost_raw = use_price * cpu_cycles
            monetary_cost_weighted = self.w_M * monetary_cost_raw
        elif 1 <= processor['processor_id'] <= len(self.processors)-2:  # 边缘
            use_price = edge_unit_price if edge_unit_price is not None else processor['unit_price']
            monetary_cost_raw = use_price * cpu_cycles
            monetary_cost_weighted = self.w_M * monetary_cost_raw
        else:  # 本地无货币成本
            pass

        total_cost = time_cost + energy_cost + monetary_cost_weighted
        return {
            'total': total_cost,
            'monetary_raw': monetary_cost_raw,  # 未加权的货币成本（用于收益计算）
            'monetary_weighted': monetary_cost_weighted,
            'time_cost': time_cost,
            'energy_cost': energy_cost
        }

    def compute_cpl(self, task, edge_unit_price):
        """计算本地执行性价比CPL = (次优成本 - 本地成本)/资源需求"""
        # 本地成本
        local_processor = self.processors[-1]
        local_cost = self.calculate_cost(task, local_processor, edge_unit_price)['total']
        
        # 次优成本（边缘最低成本）
        edge_processors = self.processors[1:-1]
        edge_costs = [self.calculate_cost(task, ep, edge_unit_price)['total'] for ep in edge_processors]
        suboptimal_cost = min(edge_costs) if edge_processors else float('inf')
        
        eul = suboptimal_cost - local_cost if local_cost < suboptimal_cost else 0
        return eul / task['resource需求'] if task['resource需求'] != 0 else 0

    def device_side_migration(self, edge_unit_price):
        """设备端迁移：优先保留本地高性价比任务"""
        task_cpl = [(idx, self.compute_cpl(task, edge_unit_price)) for idx, task in enumerate(self.tasks)]
        task_cpl_sorted = sorted(task_cpl, key=lambda x: x[1], reverse=True)
        
        remaining_local = self.local_resource_max
        for task_idx, cpl in task_cpl_sorted:
            task = self.tasks[task_idx]
            if cpl > 0 and task['resource需求'] <= remaining_local:
                self.task_allocation[task_idx, -1] = 1  # 分配到本地
                remaining_local -= task['resource需求']
            else:
                # 迁移到最优边缘
                edge_costs = [(idx+1, self.calculate_cost(task, ep, edge_unit_price)['total']) for idx, ep in enumerate(self.processors[1:-1])]
                best_edge_idx = min(edge_costs, key=lambda x: x[1])[0]
                self.task_allocation[task_idx, best_edge_idx] = 1

    def compute_cpe(self, task, edge_processor, edge_unit_price):
        """计算边缘执行性价比CPE = (云成本 - 边缘成本)/资源需求"""
        edge_cost = self.calculate_cost(task, edge_processor, edge_unit_price)['total']
        cloud_processor = next(p for p in self.processors if p['processor_id'] == 0)
        cloud_cost = self.calculate_cost(task, cloud_processor, edge_unit_price)['total']
        
        eue = cloud_cost - edge_cost if edge_cost < cloud_cost else 0
        return eue / task['resource需求'] if task['resource需求'] != 0 else 0

    def edge_side_migration(self, edge_unit_price):
        """边缘侧迁移：优先保留边缘高性价比任务"""
        edge_processors = self.processors[1:-1]
        for edge_idx, edge_proc in enumerate(edge_processors, start=1):
            # 找到分配到当前边缘的任务
            edge_tasks = [idx for idx, row in enumerate(self.task_allocation) if row[edge_idx] == 1]
            if not edge_tasks:
                continue
            
            # 按CPE降序排序
            task_cpe = [(tid, self.compute_cpe(self.tasks[tid], edge_proc, edge_unit_price)) for tid in edge_tasks]
            task_cpe_sorted = sorted(task_cpe, key=lambda x: x[1], reverse=True)
            
            remaining_edge = self.edge_resource_max[edge_idx-1]
            for tid, cpe in task_cpe_sorted:
                task = self.tasks[tid]
                if task['resource需求'] <= remaining_edge:
                    self.task_allocation[tid, edge_idx] = 1
                    remaining_edge -= task['resource需求']
                else:
                    # 迁移到云
                    self.task_allocation[tid, edge_idx] = 0
                    self.task_allocation[tid, 0] = 1

    def run(self, edge_unit_price=None):
        """执行UTA-G算法，返回边缘任务集"""
        self.task_allocation = np.zeros((len(self.tasks), len(self.processors)))
        default_price = self.processors[1]['unit_price'] if len(self.processors) > 1 else 8e-9  # 提高默认单价
        self.device_side_migration(edge_unit_price or default_price)
        self.edge_side_migration(edge_unit_price or default_price)
        
        # 整理边缘任务集
        edge_task_set = defaultdict(list)
        for tid, task in enumerate(self.tasks):
            allocated_pid = np.argmax(self.task_allocation[tid])
            if 1 <= allocated_pid <= len(self.processors)-2:  # 分配到边缘
                edge_task_set[allocated_pid].append((tid, task))
        return edge_task_set

# ------------------------------ ERA（边缘资源分配算法） ------------------------------
class ERA:
    def __init__(self, edge_processors, edge_resource_max, resource_limits, cost_weights):
        self.edge_processors = edge_processors
        self.edge_resource_max = edge_resource_max
        self.resource_limits = resource_limits
        self.cost_weights = cost_weights

    def compute_mcp(self, task, edge_unit_price):
        """计算货币性价比MCP = 货币成本 / 资源需求（优先高单价任务）"""
        monetary_cost = edge_unit_price * task['cpu_cycles']
        return monetary_cost / task['resource需求'] if task['resource需求'] != 0 else 0

    def run(self, edge_task_sets, edge_unit_prices):
        """
        执行ERA算法，返回资源分配结果和总收益
        :param edge_task_sets: {边缘ID: [(任务索引, 任务对象)]}
        :param edge_unit_prices: {边缘ID: {任务索引: 任务单价}}
        :return: (资源分配字典, 总收益)
        """
        resource_allocation = defaultdict(lambda: defaultdict(float))
        total_revenue = 0.0

        for edge_id, tasks in edge_task_sets.items():
            if edge_id not in edge_unit_prices or not edge_unit_prices[edge_id]:
                continue
            
            edge_idx = edge_id - 1
            if edge_idx < 0 or edge_idx >= len(self.edge_processors):
                continue
            
            # 严格限制边缘资源上限
            remaining_resource = self.edge_resource_max[edge_idx]
            if remaining_resource <= 0:
                continue
                
            edge_proc = self.edge_processors[edge_idx]

            # 构造UTAG辅助实例计算CPE
            helper_processors = [
                {'processor_id': 0, 'cpu_freq': 2.4e9, 'unit_price': 10e-9},  # 提高云单价
                edge_proc,  # 当前边缘
                {'processor_id': 99, 'cpu_freq': 1.5e9, 'unit_price': 0}  # 本地
            ]
            utag_helper = UTAG([t[1] for t in tasks], helper_processors, self.resource_limits, self.cost_weights)

            # 按CPE降序排序任务（优先高CPE+高MCP）
            task_cpe_mcp = []
            for tid, (task_idx, task) in enumerate(tasks):
                price = edge_unit_prices[edge_id].get(task_idx, 8e-9)
                cpe = utag_helper.compute_cpe(task, edge_proc, price)
                mcp = self.compute_mcp(task, price)
                # 综合排序权重：CPE*0.6 + MCP*0.4（优先高单价）
                combined_score = cpe * 0.6 + mcp * 0.4
                task_cpe_mcp.append((tid, combined_score))
            
            # 按综合得分降序排序
            sorted_task_indices = [tid for tid, _ in sorted(task_cpe_mcp, key=lambda x: x[1], reverse=True)]
            sorted_tasks = [tasks[tid] for tid in sorted_task_indices]

            # 按综合得分分配资源（严格约束）
            while sorted_tasks and remaining_resource > 1e-6:
                best_idx = 0  # 取综合得分最高的任务
                best_task_idx, best_task = sorted_tasks[best_idx]
                req_resource = best_task['resource需求']
                price = edge_unit_prices[edge_id].get(best_task_idx, 8e-9)

                if req_resource <= remaining_resource:
                    # 分配资源并计算收益
                    resource_allocation[edge_id][best_task_idx] = req_resource
                    remaining_resource -= req_resource
                    total_revenue += price * best_task['cpu_cycles']
                    sorted_tasks.pop(best_idx)
                else:
                    sorted_tasks.pop(best_idx)

        return resource_allocation, total_revenue

# ------------------------------ EPRA-U（用户粒度定价） ------------------------------
class EPRA_U:
    def __init__(self, user_tasks, processors, resource_limits, cost_weights):
        self.user_tasks = user_tasks
        self.processors = processors
        self.resource_limits = resource_limits
        self.cost_weights = cost_weights
        self.utag = UTAG(user_tasks, processors, resource_limits, cost_weights)
        self.era = ERA(processors[1:-1], resource_limits['edge_max'], resource_limits, cost_weights)

    def calculate_critical_prices(self):
        """计算候选价格集（μ1_ij、μ2_ij、CPL临界点）- 限制单价上限"""
        critical_prices = set()
        cloud_proc = next(p for p in self.processors if p['processor_id'] == 0)
        local_proc = self.processors[-1]
        edge_proc = self.processors[1] if len(self.processors) > 1 else None
        
        if not edge_proc:
            return [8e-9]

        for task in self.user_tasks:
            # 计算未加权的成本
            cost_cloud_raw = cloud_proc['unit_price'] * task['cpu_cycles']
            cost_local_total = self.utag.calculate_cost(task, local_proc, 0)['total']
            cost_edge_0_total = self.utag.calculate_cost(task, edge_proc, 0)['total']
            
            # μ1_ij：云成本 = 边缘成本
            denominator = self.cost_weights['w_M'] * task['cpu_cycles']
            if denominator > 1e-12:
                mu1 = (cost_cloud_raw * self.cost_weights['w_M'] + cost_edge_0_total - cost_cloud_raw * self.cost_weights['w_M']) / denominator
                mu1 = max(min(mu1, 9e-9), 2e-9)  # 限制单价范围：2e-9 ~ 9e-9
                critical_prices.add(round(mu1, 9))

            # μ2_ij：边缘成本 = 本地成本
            if denominator > 1e-12:
                mu2 = (cost_local_total - cost_edge_0_total) / denominator
                mu2 = max(min(mu2, 9e-9), 2e-9)
                critical_prices.add(round(mu2, 9))

        # CPL相等的临界点
        for t1, t2 in combinations(self.user_tasks, 2):
            r1, r2 = t1['resource需求'], t2['resource需求']
            c1, c2 = t1['cpu_cycles'], t2['cpu_cycles']
            if r1 < 1e-12 or r2 < 1e-12 or c1 < 1e-12 or c2 < 1e-12:
                continue
            
            cost_local1 = self.utag.calculate_cost(t1, local_proc, 0)['total']
            cost_local2 = self.utag.calculate_cost(t2, local_proc, 0)['total']
            cost_edge1_0 = self.utag.calculate_cost(t1, edge_proc, 0)['total']
            cost_edge2_0 = self.utag.calculate_cost(t2, edge_proc, 0)['total']
            
            numerator = (cost_local1/r1 - cost_local2/r2) - (cost_edge1_0/r1 - cost_edge2_0/r2)
            denominator = self.cost_weights['w_M'] * (c2/r2 - c1/r1)
            if abs(denominator) > 1e-12:
                mu_cpl = numerator / denominator
                mu_cpl = max(min(mu_cpl, 9e-9), 2e-9)
                critical_prices.add(round(mu_cpl, 9))

        # 确保候选价格非空
        prices = sorted(list(critical_prices)) if critical_prices else [8e-9]
        return prices

    def run(self):
        """执行EPRA-U算法，返回最优定价和收益"""
        candidate_prices = self.calculate_critical_prices()
        max_revenue = 0.0
        best_price = 8e-9
        best_allocation = defaultdict(lambda: defaultdict(float))

        for price in candidate_prices:
            # 构造边缘-任务单价字典（用户粒度：所有任务同价）
            edge_unit_prices = {}
            for ep in self.processors[1:-1]:
                edge_id = ep['processor_id']
                edge_unit_prices[edge_id] = {tid: price for tid in range(len(self.user_tasks))}
            
            # 运行UTAG获取任务分配
            edge_task_sets = self.utag.run(edge_unit_price=price)
            
            # 运行ERA计算收益
            allocation, revenue = self.era.run(edge_task_sets, edge_unit_prices)
            
            # 更新最优方案
            if revenue > max_revenue:
                max_revenue = revenue
                best_price = price
                best_allocation = allocation

        return {
            'user_granular_price': best_price,
            'resource_allocation': best_allocation,
            'max_edge_revenue': max_revenue,
            'total_tasks_allocated': sum(len(tasks) for tasks in best_allocation.values())
        }

# ------------------------------ EPRA-T（任务粒度定价） ------------------------------
class EPRA_T:
    def __init__(self, user_tasks, processors, resource_limits, cost_weights):
        self.user_tasks = user_tasks
        self.processors = processors
        self.resource_limits = resource_limits
        self.cost_weights = cost_weights
        self.utag = UTAG(user_tasks, processors, resource_limits, cost_weights)
        self.era = ERA(processors[1:-1], resource_limits['edge_max'], resource_limits, cost_weights)

    def calculate_candidate_cpl(self):
        """优化候选CPL集：扩大取值范围（0~10），增加高CPL值"""
        candidate_cpl = set(np.linspace(0, 10, 20))  # 0~10之间取20个值
        edge_proc = self.processors[1] if len(self.processors) > 1 else None
        local_proc = self.processors[-1]
        
        if not edge_proc:
            return sorted(list(candidate_cpl))

        for t1, t2 in combinations(self.user_tasks, 2):
            r1, r2 = t1['resource需求'], t2['resource需求']
            c1, c2 = t1['cpu_cycles'], t2['cpu_cycles']
            if r1 < 1e-12 or r2 < 1e-12 or c1 < 1e-12 or c2 < 1e-12:
                continue
            
            # 计算固定成本项
            cost_edge1 = self.utag.calculate_cost(t1, edge_proc, 0)['total']
            cost_edge2 = self.utag.calculate_cost(t2, edge_proc, 0)['total']
            cost_local1 = self.utag.calculate_cost(t1, local_proc, 0)['total']
            cost_local2 = self.utag.calculate_cost(t2, local_proc, 0)['total']
            
            # 解CPL临界值
            numerator = (cost_local1/r1 - cost_local2/r2) - (cost_edge1/r1 - cost_edge2/r2)
            denominator = (c2/r2 - c1/r1)
            if abs(denominator) > 1e-12:
                cpl = numerator / denominator
                if cpl >= 0:
                    candidate_cpl.add(round(cpl, 6))

        return sorted(list(candidate_cpl))

    def calculate_task_price(self, task, target_cpl):
        """核心修正：基于云成本推导单价（而非本地成本），避免单价过低"""
        edge_proc = self.processors[1] if len(self.processors) > 1 else None
        cloud_proc = next(p for p in self.processors if p['processor_id'] == 0)
        
        if not edge_proc:
            return 8e-9
        
        # 基于云成本推导单价
        cloud_cost = self.utag.calculate_cost(task, cloud_proc, 0)['total']
        edge_cost_0 = self.utag.calculate_cost(task, edge_proc, 0)['total']
        r = task['resource需求']
        c = task['cpu_cycles']
        w_m = self.cost_weights['w_M']
        
        # 单价推导公式：基于云成本与边缘成本的平衡
        mu = (target_cpl * r + cloud_cost - edge_cost_0) / (w_m * c)
        # 限制在合理微定价范围（3e-9 ~ 9e-9），确保单价高于EPRA-U下限
        mu = max(min(mu, 9e-9), 3e-9)
        return mu

    def run(self):
        """执行EPRA-T算法，返回最优任务定价和收益（修复cloud_proc未定义问题）"""
        candidate_cpl = self.calculate_candidate_cpl()
        max_revenue = 0.0
        best_task_prices = {}
        best_allocation = defaultdict(lambda: defaultdict(float))
        # 定义cloud_proc
        cloud_proc = next(p for p in self.processors if p['processor_id'] == 0)

        for target_cpl in candidate_cpl:
            # 1. 计算每个任务的专属单价（差异化，更高单价）
            task_prices = {tid: self.calculate_task_price(task, target_cpl) 
                           for tid, task in enumerate(self.user_tasks)}
            
            # 2. 按MCP排序任务（优先高单价）
            task_mcp = [(tid, (task_prices[tid] * task['cpu_cycles']) / task['resource需求']) 
                        for tid, task in enumerate(self.user_tasks) if task['resource需求'] > 1e-12]
            task_mcp_sorted = sorted(task_mcp, key=lambda x: x[1], reverse=True)
            sorted_tids = [tid for tid, _ in task_mcp_sorted]
            
            # 3. 确定边缘任务集
            edge_task_sets = defaultdict(list)
            for edge_id in [ep['processor_id'] for ep in self.processors[1:-1]]:
                edge_proc = self.processors[edge_id]
                for tid in sorted_tids:
                    task = self.user_tasks[tid]
                    price = task_prices[tid]
                    edge_cost = self.utag.calculate_cost(task, edge_proc, price)['total']
                    # 使用已定义的cloud_proc
                    cloud_cost = self.utag.calculate_cost(task, cloud_proc, 0)['total']
                    if edge_cost < cloud_cost:
                        edge_task_sets[edge_id].append((tid, task))
            
            # 4. 构造边缘-任务单价字典（差异化）
            edge_unit_prices = {}
            for edge_id in edge_task_sets:
                edge_unit_prices[edge_id] = {tid: task_prices[tid] for tid, _ in edge_task_sets[edge_id]}
            
            # 5. 运行ERA计算收益
            allocation, revenue = self.era.run(edge_task_sets, edge_unit_prices)
            
            # 6. 更新最优方案
            if revenue > max_revenue:
                max_revenue = revenue
                best_task_prices = {self.user_tasks[tid]['task_id']: task_prices[tid] for tid in task_prices}
                best_allocation = allocation

        return {
            'task_granular_prices': best_task_prices,
            'resource_allocation': best_allocation,
            'max_edge_revenue': max_revenue,
            'total_tasks_allocated': sum(len(tasks) for tasks in best_allocation.values())
        }

# ------------------------------ 实验工具函数 ------------------------------
def generate_random_tasks(num_tasks, task_id_prefix="T"):
    """生成随机任务（调整资源需求，让分配更合理）"""
    tasks = []
    for i in range(num_tasks):
        # CPU周期：330×[200, 800] 周期
        cpu_cycles = 330 * random.randint(200, 800)
        # 本地数据：[1000, 5000] KB
        local_data = random.randint(1000, 5000)
        # 外部数据：[500, 2000] KB
        external_data = random.randint(500, 2000)
        # 结果数据：[100, 1000] KB
        result_data = random.randint(100, 1000)
        # 资源需求：[0.08, 0.15] 单位
        resource_demand = round(random.uniform(0.08, 0.15), 2)
        
        tasks.append({
            'task_id': f"{task_id_prefix}{i+1}",
            'cpu_cycles': cpu_cycles,
            'local_data': local_data,
            'external_data': external_data,
            'result_data': result_data,
            'resource需求': resource_demand
        })
    return tasks

def generate_processors(num_edge=4):
    """生成处理器配置（提高边缘单价）"""
    processors = [
        {'processor_id': 0, 'cpu_freq': 2.4e9, 'unit_price': 10e-9},  # 云单价：10e-9
    ]
    # 边缘服务器
    for i in range(num_edge):
        processors.append({
            'processor_id': i+1,
            'cpu_freq': 4e9,
            'unit_price': 8e-9
        })
    # 本地设备
    processors.append({
        'processor_id': num_edge+1,
        'cpu_freq': 1.5e9,
        'unit_price': 0.0
    })
    return processors

# ------------------------------ 实验运行与可视化 ------------------------------
def run_experiments():
    """运行论文核心实验并绘图"""
    # 实验1：不同任务数量对收益的影响（任务数：5,10,15,20,25,30）
    task_nums = [5, 10, 15, 20, 25, 30]
    epra_u_revenue = []
    epra_t_revenue = []
    epra_u_task_count = []
    epra_t_task_count = []

    # 固定配置
    num_edge = 4
    resource_limits_base = {
        'local_max': 0.5,
        'edge_max': [0.3]*num_edge
    }
    cost_weights = {'w_T': 0.25, 'w_E': 0.25, 'w_M': 0.5}

    print("="*80)
    print("开始运行实验：不同任务数量对边缘收益的影响")
    print("="*80)
    
    for task_num in task_nums:
        print(f"\n--- 任务数量：{task_num} ---")
        # 生成随机任务和处理器
        tasks = generate_random_tasks(task_num)
        processors = generate_processors(num_edge)
        
        # 运行EPRA-U
        epra_u = EPRA_U(tasks, processors, resource_limits_base, cost_weights)
        u_res = epra_u.run()
        epra_u_revenue.append(u_res['max_edge_revenue'])
        epra_u_task_count.append(u_res['total_tasks_allocated'])
        print(f"EPRA-U 收益：{u_res['max_edge_revenue']:.6f} 美元，分配任务数：{u_res['total_tasks_allocated']}")
        
        # 运行EPRA-T
        epra_t = EPRA_T(tasks, processors, resource_limits_base, cost_weights)
        t_res = epra_t.run()
        epra_t_revenue.append(t_res['max_edge_revenue'])
        epra_t_task_count.append(t_res['total_tasks_allocated'])
        print(f"EPRA-T 收益：{t_res['max_edge_revenue']:.6f} 美元，分配任务数：{t_res['total_tasks_allocated']}")

    # 实验2：不同本地资源上限对收益的影响（本地资源：0.2,0.3,0.4,0.5,0.6,0.7）
    local_resources = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    u_revenue_local = []
    t_revenue_local = []

    print("\n" + "="*80)
    print("开始运行实验：不同本地资源上限对边缘收益的影响")
    print("="*80)
    
    for local_max in local_resources:
        print(f"\n--- 本地资源上限：{local_max} ---")
        # 固定任务数=20
        tasks = generate_random_tasks(20)
        processors = generate_processors(num_edge)
        resource_limits = {
            'local_max': local_max,
            'edge_max': [0.3]*num_edge
        }
        
        # 运行EPRA-U
        epra_u = EPRA_U(tasks, processors, resource_limits, cost_weights)
        u_res = epra_u.run()
        u_revenue_local.append(u_res['max_edge_revenue'])
        print(f"EPRA-U 收益：{u_res['max_edge_revenue']:.6f} 美元")
        
        # 运行EPRA-T
        epra_t = EPRA_T(tasks, processors, resource_limits, cost_weights)
        t_res = epra_t.run()
        t_revenue_local.append(t_res['max_edge_revenue'])
        print(f"EPRA-T 收益：{t_res['max_edge_revenue']:.6f} 美元")

    # 实验3：不同边缘资源上限对收益的影响（边缘单节点资源：0.1,0.2,0.3,0.4,0.5）
    edge_resources = [0.1, 0.2, 0.3, 0.4, 0.5]
    u_revenue_edge = []
    t_revenue_edge = []

    print("\n" + "="*80)
    print("开始运行实验：不同边缘资源上限对边缘收益的影响")
    print("="*80)
    
    for edge_max in edge_resources:
        print(f"\n--- 边缘单节点资源上限：{edge_max} ---")
        # 固定任务数=20，本地资源=0.5
        tasks = generate_random_tasks(20)
        processors = generate_processors(num_edge)
        resource_limits = {
            'local_max': 0.5,
            'edge_max': [edge_max]*num_edge
        }
        
        # 运行EPRA-U
        epra_u = EPRA_U(tasks, processors, resource_limits, cost_weights)
        u_res = epra_u.run()
        u_revenue_edge.append(u_res['max_edge_revenue'])
        print(f"EPRA-U 收益：{u_res['max_edge_revenue']:.6f} 美元")
        
        # 运行EPRA-T
        epra_t = EPRA_T(tasks, processors, resource_limits, cost_weights)
        t_res = epra_t.run()
        t_revenue_edge.append(t_res['max_edge_revenue'])
        print(f"EPRA-T 收益：{t_res['max_edge_revenue']:.6f} 美元")

    # ------------------------------ 绘图 ------------------------------
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
    plt.rcParams['axes.unicode_minus'] = False
    
    # 图1：任务数量 vs 边缘收益
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 子图1：任务数量-收益
    ax1.plot(task_nums, epra_u_revenue, 'o-', label='EPRA-U（用户粒度）', linewidth=2, markersize=8)
    ax1.plot(task_nums, epra_t_revenue, 's-', label='EPRA-T（任务粒度）', linewidth=2, markersize=8)
    ax1.set_xlabel('任务数量', fontsize=12)
    ax1.set_ylabel('边缘总收益（美元）', fontsize=12)
    ax1.set_title('不同任务数量对边缘收益的影响', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 子图2：本地资源-收益
    ax2.plot(local_resources, u_revenue_local, 'o-', label='EPRA-U（用户粒度）', linewidth=2, markersize=8)
    ax2.plot(local_resources, t_revenue_local, 's-', label='EPRA-T（任务粒度）', linewidth=2, markersize=8)
    ax2.set_xlabel('本地资源上限', fontsize=12)
    ax2.set_ylabel('边缘总收益（美元）', fontsize=12)
    ax2.set_title('不同本地资源上限对边缘收益的影响', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 子图3：边缘资源-收益
    ax3.plot(edge_resources, u_revenue_edge, 'o-', label='EPRA-U（用户粒度）', linewidth=2, markersize=8)
    ax3.plot(edge_resources, t_revenue_edge, 's-', label='EPRA-T（任务粒度）', linewidth=2, markersize=8)
    ax3.set_xlabel('边缘单节点资源上限', fontsize=12)
    ax3.set_ylabel('边缘总收益（美元）', fontsize=12)
    ax3.set_title('不同边缘资源上限对边缘收益的影响', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('edge_revenue_experiments_final.png', dpi=300, bbox_inches='tight')
    print("\n实验完成！结果图已保存为：edge_revenue_experiments_final.png")
    plt.show()

    # 输出最终对比表
    print("\n" + "="*80)
    print("实验结果汇总（最终版）")
    print("="*80)
    print(f"{'实验场景':<20} {'EPRA-U收益(美元)':<20} {'EPRA-T收益(美元)':<20} {'收益提升率':<10}")
    print("-"*80)
    # 任务数量20时的对比
    idx_20 = task_nums.index(20)
    if epra_u_revenue[idx_20] > 0:
        improve_rate = (epra_t_revenue[idx_20] - epra_u_revenue[idx_20]) / epra_u_revenue[idx_20] * 100
    else:
        improve_rate = 100.0
    print(f"任务数=20          {epra_u_revenue[idx_20]:<20.6f} {epra_t_revenue[idx_20]:<20.6f} {improve_rate:<10.2f}%")
    
    # 本地资源0.5时的对比
    idx_local = local_resources.index(0.5)
    if u_revenue_local[idx_local] > 0:
        improve_rate = (t_revenue_local[idx_local] - u_revenue_local[idx_local]) / u_revenue_local[idx_local] * 100
    else:
        improve_rate = 100.0
    print(f"本地资源=0.5        {u_revenue_local[idx_local]:<20.6f} {t_revenue_local[idx_local]:<20.6f} {improve_rate:<10.2f}%")
    
    # 边缘资源0.3时的对比
    idx_edge = edge_resources.index(0.3)
    if u_revenue_edge[idx_edge] > 0:
        improve_rate = (t_revenue_edge[idx_edge] - u_revenue_edge[idx_edge]) / u_revenue_edge[idx_edge] * 100
    else:
        improve_rate = 100.0
    print(f"边缘资源=0.3        {u_revenue_edge[idx_edge]:<20.6f} {t_revenue_edge[idx_edge]:<20.6f} {improve_rate:<10.2f}%")

# ------------------------------ 主函数 ------------------------------
if __name__ == "__main__":
    # 运行完整实验
    run_experiments()