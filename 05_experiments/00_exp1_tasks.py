# experiments/exp1_tasks.py

"""
实验 1：任务数量变化对边缘收益的影响
"""

from 03_models.01_user_device import UserDevice
from 03_models.00_task import Task
from 02_algorithms.00_epra_u import epra_u
from 02_algorithms.01_epra_t import epra_t
from 02_algorithms.03_baseline import baseline_local, baseline_cloud, baseline_min_cost, baseline_random
from 04_utils.01_plot_utils import draw_six_lines
from 05_experiments.04_helper_generate import generate_user, build_edge_cloud


def run_experiment_1():
    x_vals = [5, 10, 15, 20, 25, 30]

    yU=[]; yT=[]; yL=[]; yC=[]; yMC=[]; yR=[]

    print("=============================================")
    print(" 实验1：任务数量 对 边缘收益 的影响")
    print("=============================================\n")

    for n in x_vals:
        # 生成用户 + 任务
        user = generate_user(user_id=0, num_tasks=n)

        # 系统配置
        edge, cloud = build_edge_cloud()

        # EPRA-U
        mu_list = [x * 1e-11 for x in range(5, 21, 3)]
        _, rev_u = epra_u(user, edge, cloud, mu_list)

        # EPRA-T
        cpl_list = [0.0, 1e-3, 5e-3, 1e-2]
        _, rev_t = epra_t(user, edge, cloud, cpl_list, 1e-11, 5e-11)

        # Baselines
        rev_l  = baseline_local(user, edge, cloud)
        rev_c  = baseline_cloud(user, edge, cloud)
        rev_mc = baseline_min_cost(user, edge, cloud)
        rev_r  = baseline_random(user, edge, cloud)

        yU.append(rev_u); yT.append(rev_t)
        yL.append(rev_l); yC.append(rev_c)
        yMC.append(rev_mc); yR.append(rev_r)

        # B版科研输出
        print(f"任务数 = {n}:")
        print(f"    EPRA-U 收益     = {rev_u:.4e}")
        print(f"    EPRA-T 收益     = {rev_t:.4e}")
        print(f"    Local-only 收益 = {rev_l:.4e}")
        print(f"    Cloud-only 收益 = {rev_c:.4e}")
        print(f"    Min-Cost 收益   = {rev_mc:.4e}")
        print(f"    Random 收益     = {rev_r:.4e}")
        print("---------------------------------------------")

    draw_six_lines(
        x_vals, yU, yT, yL, yC, yMC, yR,
        xlabel="任务数量",
        ylabel="边缘收益",
        title="任务数量对边缘收益的影响",
        filename="exp1_sixlines.png"
    )

