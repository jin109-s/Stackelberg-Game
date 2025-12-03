# utils/plot_utils.py

"""
绘图工具模块：
六条彩色折线图（EPRA-U / EPRA-T / Local / Cloud / MinCost / Random）
"""

import matplotlib.pyplot as plt

# 确保中文不乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def draw_six_lines(x, EPRAU, EPRAT, Local, Cloud, MinCost, Random,
                   xlabel, ylabel, title, filename):
    """
    绘制论文级彩色六折线图
    """

    plt.figure(figsize=(7,5))

    plt.plot(x, EPRAU, marker='o', color='blue', label='EPRA-U')
    plt.plot(x, EPRAT, marker='s', color='orange', label='EPRA-T')
    plt.plot(x, Local, marker='^', color='green', label='Local-only')
    plt.plot(x, Cloud, marker='v', color='purple', label='Cloud-only')
    plt.plot(x, MinCost, marker='>', color='red', label='Min-Cost')
    plt.plot(x, Random, marker='<', color='gray', label='Random')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)

    print(f"图已保存：{filename}")

