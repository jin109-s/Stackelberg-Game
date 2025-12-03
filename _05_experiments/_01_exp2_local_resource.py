# experiments/exp2_local_resource.py

"""
实验2：本地资源系数 对 边缘收益 的影响
"""

from _02_algorithms._00_epra_u import epra_u
from _02_algorithms._01_epra_t import epra_t
from _02_algorithms._03_baseline import baseline_local, baseline_cloud, baseline_min_cost, baseline_random
from _04_utils._01_plot_utils import draw_six_lines
from _05_experiments._04_helper_generate import generate_user, build_edge_cloud


def run_experiment_2():
    x_vals = [0.2, 0.4, 0.6, 0.8, 1.0]

    yU=[]; yT=[]; yL=[]; yC=[]; yMC=[]; yR=[]

    print("=============================================")
    print(" 实验2：本地资源系数 对 边缘收益 的影响")
    print("=============================================\n")

    for factor in x_vals:
        user = generate_user(user_id=0, num_tasks=20, local_factor=factor)
        edge, cloud = build_edge_cloud()

        mu_list = [x * 1e-11 for x in range(5, 21, 3)]
        _, rev_u = epra_u(user, edge, cloud, mu_list)

        cpl_list = [0.0, 1e-3, 5e-3, 1e-2]
        _, rev_t = epra_t(user, edge, cloud, cpl_list, 1e-11, 5e-11)

        rev_l  = baseline_local(user, edge, cloud)
        rev_c  = baseline_cloud(user, edge, cloud)
        rev_mc = baseline_min_cost(user, edge, cloud)
        rev_r  = baseline_random(user, edge, cloud)

        yU.append(rev_u); yT.append(rev_t)
        yL.append(rev_l); yC.append(rev_c)
        yMC.append(rev_mc); yR.append(rev_r)

        print(f"本地资源系数 = {factor:.1f}:")
        print(f"    EPRA-U 收益     = {rev_u:.4e}")
        print(f"    EPRA-T 收益     = {rev_t:.4e}")
        print(f"    Local-only 收益 = {rev_l:.4e}")
        print(f"    Cloud-only 收益 = {rev_c:.4e}")
        print(f"    Min-Cost 收益   = {rev_mc:.4e}")
        print(f"    Random 收益     = {rev_r:.4e}")
        print("---------------------------------------------")

    draw_six_lines(
        x_vals, yU, yT, yL, yC, yMC, yR,
        xlabel="本地资源系数（相对）",
        ylabel="边缘收益",
        title="本地资源对边缘收益的影响",
        filename="exp2_sixlines.png"
    )


