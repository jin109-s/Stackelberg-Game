import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

# ============================
# 全局中文 & 随机种子
# ============================
plt.rcParams['font.sans-serif'] = ['SimHei']   # 支持中文
plt.rcParams['axes.unicode_minus'] = False     # 负号正常显示
random.seed(0)

# ============================
# 数据结构
# ============================

@dataclass
class Task:
    user_id: int
    task_id: int
    Cij: float
    Din: float
    Dout: float
    Rij: float
    wT: float
    wE: float
    wM: float

@dataclass
class UserDevice:
    user_id: int
    freq: float
    beta: float
    local_resource_max: float
    tasks: List[Task] = field(default_factory=list)

@dataclass
class EdgeServer:
    server_id: int
    freq: float
    total_resource: float
    per_user_resource_max: Dict[int, float] = field(default_factory=dict)

@dataclass
class CloudServer:
    freq: float
    mu_c: float

# ============================
# 系统参数
# ============================

B_UP_EDGE = 5000.0
B_DOWN_EDGE = 5000.0
B_UP_CLOUD = 1000.0
B_DOWN_CLOUD = 1000.0

P_TX = 0.1
P_RX = 0.05

# ============================
# 基础函数
# ============================

def uplink_time(data_kb, bandwidth):
    return data_kb / bandwidth

def downlink_time(data_kb, bandwidth):
    return data_kb / bandwidth

def uplink_energy(t):
    return P_TX * t

def downlink_energy(t):
    return P_RX * t

# ============================
# 成本函数
# ============================

def cost_local(task, user):
    T_comp = task.Cij / user.freq
    E_comp = user.beta * task.Cij * (user.freq ** 2)
    return task.wT * T_comp + task.wE * E_comp

def cost_edge(task, user, edge, mu):
    t_up = uplink_time(task.Din, B_UP_EDGE)
    t_down = downlink_time(task.Dout, B_DOWN_EDGE)
    T_comm = t_up + t_down
    E_comm = uplink_energy(t_up) + downlink_energy(t_down)
    T_comp = task.Cij / edge.freq
    return task.wT * (T_comm + T_comp) + task.wE * E_comm + task.wM * (mu * task.Cij)

def cost_cloud(task, cloud):
    t_up = uplink_time(task.Din, B_UP_CLOUD)
    t_down = downlink_time(task.Dout, B_DOWN_CLOUD)
    T_comm = t_up + t_down
    E_comm = uplink_energy(t_up) + downlink_energy(t_down)
    T_comp = task.Cij / cloud.freq
    return task.wT * (T_comm + T_comp) + task.wE * E_comm + task.wM * (cloud.mu_c * task.Cij)

# ============================
# UTA-G 用户决策
# ============================

def uta_g_single_user(user, edges, cloud, mu_edge):
    decisions = {}
    edge = edges[0]

    Uvals = {}
    best_place = {}
    for t in user.tasks:
        key = (t.user_id, t.task_id)
        mu = mu_edge[key]
        Ul = cost_local(t, user)
        Ue = cost_edge(t, user, edge, mu)
        Uc = cost_cloud(t, cloud)
        Uvals[key] = (Ul, Ue, Uc)
        best_place[key] = min([(Ul, 'local'), (Ue, 'edge'), (Uc, 'cloud')])[1]

    # 本地资源检查
    local_tasks = [t for t in user.tasks if best_place[(t.user_id, t.task_id)] == 'local']
    total_local = sum(t.Rij for t in local_tasks)
    stay_local = set()
    mig_local = set()

    if total_local <= user.local_resource_max:
        for t in local_tasks:
            stay_local.add((t.user_id, t.task_id))
    else:
        CPL_list = []
        for t in local_tasks:
            key = (t.user_id, t.task_id)
            Ul, Ue, Uc = Uvals[key]
            CPL = (min(Ue, Uc) - Ul) / t.Rij
            CPL_list.append((CPL, t))
        CPL_list.sort(key=lambda x: x[0], reverse=True)
        remaining = user.local_resource_max
        for cpl, t in CPL_list:
            if t.Rij <= remaining and cpl > 0:
                stay_local.add((t.user_id, t.task_id))
                remaining -= t.Rij
            else:
                mig_local.add((t.user_id, t.task_id))

    # 本地决策
    for key in stay_local:
        decisions[key] = 'local'

    # 从本地迁走的
    for key in mig_local:
        uid, tid = key
        t = next(x for x in user.tasks if x.task_id == tid)
        mu = mu_edge[key]
        Ue = cost_edge(t, user, edge, mu)
        Uc = cost_cloud(t, cloud)
        decisions[key] = 'edge_0' if Ue <= Uc else 'cloud'

    # 初始不在本地的
    for t in user.tasks:
        key = (t.user_id, t.task_id)
        if key not in decisions:
            decisions[key] = 'edge_0' if best_place[key] == 'edge' else best_place[key]

    # 边缘资源检查
    edge_tasks = [t for t in user.tasks if decisions[(t.user_id, t.task_id)].startswith('edge')]
    Rmax = edge.per_user_resource_max.get(user.user_id, edge.total_resource)
    need = sum(t.Rij for t in edge_tasks)

    if need <= Rmax:
        return decisions

    CPE_list = []
    for t in edge_tasks:
        key = (t.user_id, t.task_id)
        _, Ue, Uc = Uvals[key]
        CPE = (Uc - Ue) / t.Rij
        CPE_list.append((CPE, t))
    CPE_list.sort(key=lambda x: x[0], reverse=True)

    remaining = Rmax
    stay_edge = set()
    mig_cloud = set()
    for cpe, t in CPE_list:
        key = (t.user_id, t.task_id)
        if t.Rij <= remaining and cpe > 0:
            stay_edge.add(key)
            remaining -= t.Rij
        else:
            mig_cloud.add(key)
    for key in mig_cloud:
        decisions[key] = 'cloud'

    return decisions

# ============================
# ERA 边缘资源分配
# ============================

def era_allocation(edge, user, task_set, mu_edge):
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

# ============================
# EPRA-U / EPRA-T
# ============================

def epra_u(user, edge, cloud, mu_list):
    best_mu = mu_list[0]
    best_rev = 0.0
    for mu in mu_list:
        mu_edge = {(t.user_id, t.task_id): mu for t in user.tasks}
        decisions = uta_g_single_user(user, [edge], cloud, mu_edge)
        edge_tasks = [t for t in user.tasks if decisions[(t.user_id, t.task_id)].startswith('edge')]
        rev = era_allocation(edge, user, edge_tasks, mu_edge)
        if rev > best_rev:
            best_rev = rev
            best_mu = mu
    return best_mu, best_rev

def epra_t(user, edge, cloud, cpl_list, mu_min, mu_max):
    best_rev = 0.0
    best_mu_dict = {}
    for cpl in cpl_list:
        mu_edge = {}
        for t in user.tasks:
            key = (t.user_id, t.task_id)
            Ul = cost_local(t, user)
            U0 = cost_edge(t, user, edge, 0.0)
            target = Ul + cpl * t.Rij
            if t.wM * t.Cij > 0:
                mu = (target - U0) / (t.wM * t.Cij)
            else:
                mu = mu_min
            mu = max(mu_min, min(mu_max, mu))
            mu_edge[key] = mu

        decisions = uta_g_single_user(user, [edge], cloud, mu_edge)
        edge_tasks = [t for t in user.tasks if decisions[(t.user_id, t.task_id)].startswith('edge')]
        rev = era_allocation(edge, user, edge_tasks, mu_edge)

        if rev > best_rev:
            best_rev = rev
            best_mu_dict = mu_edge.copy()
    return best_mu_dict, best_rev

# ============================
# 四个 baseline
# ============================

def baseline_local(user, edge, cloud):
    return 0.0   # 全本地执行，不经过边缘

def baseline_cloud(user, edge, cloud):
    return 0.0   # 全云执行，不经过边缘

def baseline_min_cost(user, edge, cloud):
    decisions = {}
    for t in user.tasks:
        Ul = cost_local(t, user)
        Ue = cost_edge(t, user, edge, 0.0)
        Uc = cost_cloud(t, cloud)
        place = min([(Ul, 'local'), (Ue, 'edge'), (Uc, 'cloud')])[1]
        decisions[(t.user_id, t.task_id)] = place
    edge_tasks = [t for t in user.tasks if decisions[(t.user_id, t.task_id)] == 'edge']
    mu_edge = {(t.user_id, t.task_id): 1e-11 for t in edge_tasks}
    revenue = sum(mu_edge[(t.user_id, t.task_id)] * t.Cij for t in edge_tasks)
    return revenue

def baseline_random(user, edge, cloud):
    decisions = {}
    choices = ['local', 'edge', 'cloud']
    for t in user.tasks:
        decisions[(t.user_id, t.task_id)] = random.choice(choices)
    edge_tasks = [t for t in user.tasks if decisions[(t.user_id, t.task_id)] == 'edge']
    mu_edge = {(t.user_id, t.task_id): 1e-11 for t in edge_tasks}
    revenue = sum(mu_edge[(t.user_id, t.task_id)] * t.Cij for t in edge_tasks)
    return revenue

# ============================
# 任务生成 & 系统构建
# ============================

def generate_random_user(uid, num_tasks, local_factor=1.0, input_scale=1.0):
    freq = 1.5e9
    beta = 1e-27
    base_res = 150.0
    user = UserDevice(uid, freq, beta, base_res * local_factor)
    for tid in range(num_tasks):
        C = random.uniform(1e8, 5e8)
        Din = random.uniform(500, 2000) * input_scale
        Dout = random.uniform(100, 500) * input_scale
        Rij = random.uniform(5, 20)
        wM = random.uniform(0.2, 0.8)
        rem = 1 - wM
        wT = random.uniform(rem * 0.4, rem * 0.6)
        wE = rem - wT
        user.tasks.append(Task(uid, tid, C, Din, Dout, Rij, wT, wE, wM))
    return user

def build_edge_cloud(edge_factor=1.0):
    base_total = 500.0
    per_user = 200.0
    edge = EdgeServer(0, 4e9, base_total * edge_factor)
    edge.per_user_resource_max[0] = per_user * edge_factor
    cloud = CloudServer(2.4e9, 6e-12)
    return edge, cloud

# ============================
# 画六折线彩色图
# ============================

def draw_six_lines(x, EPRAU, EPRAT, L, C, MINC, R,
                   xlabel, ylabel, title, filename):
    plt.figure(figsize=(7,5))
    plt.plot(x, EPRAU, marker='o', color='blue',   label='EPRA-U')
    plt.plot(x, EPRAT, marker='s', color='orange', label='EPRA-T')
    plt.plot(x, L,     marker='^', color='green',  label='Local-only')
    plt.plot(x, C,     marker='v', color='purple', label='Cloud-only')
    plt.plot(x, MINC,  marker='>', color='red',    label='Min-Cost')
    plt.plot(x, R,     marker='<', color='gray',   label='Random')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    print(f"图 已保存为：{filename}\n")

# ============================
# 四组实验（带 B 版科研输出）
# ============================

def experiment_1():
    x_vals = [5, 10, 15, 20, 25, 30]
    yU=[]; yT=[]; yL=[]; yC=[]; yMC=[]; yR=[]

    print("===========================================")
    print(" 实验 1：任务数量 对 边缘收益 的影响")
    print("===========================================\n")

    for n in x_vals:
        user = generate_random_user(0, n)
        edge, cloud = build_edge_cloud()

        mu_list = [x * 1e-11 for x in range(5, 21, 3)]
        _, rev_u = epra_u(user, edge, cloud, mu_list)

        cpl_list = [0.0, 1e-3, 5e-3, 1e-2]
        _, rev_t = epra_t(user, edge, cloud, cpl_list, 1e-11, 5e-11)

        rev_l  = baseline_local(user, edge, cloud)
        rev_c  = baseline_cloud(user, edge, cloud)
        rev_mc = baseline_min_cost(user, edge, cloud)
        rev_r  = baseline_random(user, edge, cloud)

        yU.append(rev_u)
        yT.append(rev_t)
        yL.append(rev_l)
        yC.append(rev_c)
        yMC.append(rev_mc)
        yR.append(rev_r)

        print(f"任务数 = {n}:")
        print(f"    EPRA-U 收益       = {rev_u:.4e}")
        print(f"    EPRA-T 收益       = {rev_t:.4e}")
        print(f"    Local-only 收益   = {rev_l:.4e}")
        print(f"    Cloud-only 收益   = {rev_c:.4e}")
        print(f"    Min-Cost 收益     = {rev_mc:.4e}")
        print(f"    Random 收益       = {rev_r:.4e}")
        print("-------------------------------------------")

    draw_six_lines(x_vals, yU, yT, yL, yC, yMC, yR,
                   "任务数量", "边缘收益",
                   "任务数量对边缘收益的影响",
                   "exp1_sixlines.png")

def experiment_2():
    x_vals = [0.2, 0.4, 0.6, 0.8, 1.0]
    yU=[]; yT=[]; yL=[]; yC=[]; yMC=[]; yR=[]

    print("===========================================")
    print(" 实验 2：本地资源系数 对 边缘收益 的影响")
    print("===========================================\n")

    for f in x_vals:
        user = generate_random_user(0, 20, local_factor=f)
        edge, cloud = build_edge_cloud()

        mu_list = [x * 1e-11 for x in range(5, 21, 3)]
        _, rev_u = epra_u(user, edge, cloud, mu_list)

        cpl_list = [0.0, 1e-3, 5e-3, 1e-2]
        _, rev_t = epra_t(user, edge, cloud, cpl_list, 1e-11, 5e-11)

        rev_l  = baseline_local(user, edge, cloud)
        rev_c  = baseline_cloud(user, edge, cloud)
        rev_mc = baseline_min_cost(user, edge, cloud)
        rev_r  = baseline_random(user, edge, cloud)

        yU.append(rev_u)
        yT.append(rev_t)
        yL.append(rev_l)
        yC.append(rev_c)
        yMC.append(rev_mc)
        yR.append(rev_r)

        print(f"本地资源系数 = {f:.1f}:")
        print(f"    EPRA-U 收益       = {rev_u:.4e}")
        print(f"    EPRA-T 收益       = {rev_t:.4e}")
        print(f"    Local-only 收益   = {rev_l:.4e}")
        print(f"    Cloud-only 收益   = {rev_c:.4e}")
        print(f"    Min-Cost 收益     = {rev_mc:.4e}")
        print(f"    Random 收益       = {rev_r:.4e}")
        print("-------------------------------------------")

    draw_six_lines(x_vals, yU, yT, yL, yC, yMC, yR,
                   "本地资源系数（相对）", "边缘收益",
                   "本地资源对边缘收益的影响",
                   "exp2_sixlines.png")

def experiment_3():
    x_vals = [0.2, 0.4, 0.6, 0.8, 1.0]
    yU=[]; yT=[]; yL=[]; yC=[]; yMC=[]; yR=[]

    print("===========================================")
    print(" 实验 3：边缘资源系数 对 边缘收益 的影响")
    print("===========================================\n")

    for f in x_vals:
        user = generate_random_user(0, 20)
        edge, cloud = build_edge_cloud(edge_factor=f)

        mu_list = [x * 1e-11 for x in range(5, 21, 3)]
        _, rev_u = epra_u(user, edge, cloud, mu_list)

        cpl_list = [0.0, 1e-3, 5e-3, 1e-2]
        _, rev_t = epra_t(user, edge, cloud, cpl_list, 1e-11, 5e-11)

        rev_l  = baseline_local(user, edge, cloud)
        rev_c  = baseline_cloud(user, edge, cloud)
        rev_mc = baseline_min_cost(user, edge, cloud)
        rev_r  = baseline_random(user, edge, cloud)

        yU.append(rev_u)
        yT.append(rev_t)
        yL.append(rev_l)
        yC.append(rev_c)
        yMC.append(rev_mc)
        yR.append(rev_r)

        print(f"边缘资源系数 = {f:.1f}:")
        print(f"    EPRA-U 收益       = {rev_u:.4e}")
        print(f"    EPRA-T 收益       = {rev_t:.4e}")
        print(f"    Local-only 收益   = {rev_l:.4e}")
        print(f"    Cloud-only 收益   = {rev_c:.4e}")
        print(f"    Min-Cost 收益     = {rev_mc:.4e}")
        print(f"    Random 收益       = {rev_r:.4e}")
        print("-------------------------------------------")

    draw_six_lines(x_vals, yU, yT, yL, yC, yMC, yR,
                   "边缘资源系数（相对）", "边缘收益",
                   "边缘资源对边缘收益的影响",
                   "exp3_sixlines.png")

def experiment_4():
    x_vals = [0.5, 1.0, 1.5, 2.0, 2.5]
    yU=[]; yT=[]; yL=[]; yC=[]; yMC=[]; yR=[]

    print("===========================================")
    print(" 实验 4：输入数据大小倍数 对 边缘收益 的影响")
    print("===========================================\n")

    for s in x_vals:
        user = generate_random_user(0, 20, input_scale=s)
        edge, cloud = build_edge_cloud()

        mu_list = [x * 1e-11 for x in range(5, 21, 3)]
        _, rev_u = epra_u(user, edge, cloud, mu_list)

        cpl_list = [0.0, 1e-3, 5e-3, 1e-2]
        _, rev_t = epra_t(user, edge, cloud, cpl_list, 1e-11, 5e-11)

        rev_l  = baseline_local(user, edge, cloud)
        rev_c  = baseline_cloud(user, edge, cloud)
        rev_mc = baseline_min_cost(user, edge, cloud)
        rev_r  = baseline_random(user, edge, cloud)

        yU.append(rev_u)
        yT.append(rev_t)
        yL.append(rev_l)
        yC.append(rev_c)
        yMC.append(rev_mc)
        yR.append(rev_r)

        print(f"输入数据放大倍数 = {s:.1f}:")
        print(f"    EPRA-U 收益       = {rev_u:.4e}")
        print(f"    EPRA-T 收益       = {rev_t:.4e}")
        print(f"    Local-only 收益   = {rev_l:.4e}")
        print(f"    Cloud-only 收益   = {rev_c:.4e}")
        print(f"    Min-Cost 收益     = {rev_mc:.4e}")
        print(f"    Random 收益       = {rev_r:.4e}")
        print("-------------------------------------------")

    draw_six_lines(x_vals, yU, yT, yL, yC, yMC, yR,
                   "输入数据放大倍数", "边缘收益",
                   "输入数据大小对边缘收益的影响",
                   "exp4_sixlines.png")

# ============================
# 主程序
# ============================

if __name__ == "__main__":
    print("开始运行四组六折线实验……\n")
    experiment_1()
    experiment_2()
    experiment_3()
    experiment_4()
    print("全部实验完成，图像与结果已生成。")
