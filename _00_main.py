# main.py

"""
EPRA-Stackelberg-Replication
一键运行全部四个实验的主程序
"""

from _05_experiments._00_exp1_tasks import run_experiment_1
from _05_experiments._01_exp2_local_resource import run_experiment_2
from _05_experiments._02_exp3_edge_resource import run_experiment_3
from _05_experiments._03_exp4_input_scale import run_experiment_4


def main():

    print("\n==========================================")
    print("   EPRA Stackelberg 论文复现（六折线版本）")
    print("==========================================\n")

    print(">>> 开始运行实验1（任务数量变化）...\n")
    run_experiment_1()
    print(">>> 实验1完成。\n")

    print(">>> 开始运行实验2（本地资源系数变化）...\n")
    run_experiment_2()
    print(">>> 实验2完成。\n")

    print(">>> 开始运行实验3（边缘资源扩容）...\n")
    run_experiment_3()
    print(">>> 实验3完成。\n")

    print(">>> 开始运行实验4（输入数据大小变化）...\n")
    run_experiment_4()
    print(">>> 实验4完成。\n")

    print("==========================================")
    print("   所有实验已运行完毕，图像已生成！")
    print("==========================================\n")


if __name__ == "__main__":
    main()


