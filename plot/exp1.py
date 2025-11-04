import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import os
# 设置中文字体（解决中文显示问题）
# 1. 确保字体文件存在

# 查找中文字体路径
zh_font_path = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"  # WenQuanYi Micro Hei
zh_font = fm.FontProperties(fname=zh_font_path)

# 设置中文字体和解决负号显示问题
plt.rcParams['font.sans-serif'] = [zh_font.get_name()]  # 使用指定的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

plt.rcParams.update({'font.size': 20})  # 默认字体大小
# 数据准备：合并所有实验数据
data_qp = [
    {"N": 1000, "type": "QP", "avg_encode_latency_ns": 1.500700, "p95_encode_latency_ns": 2.096000, "avg_decode_latency_ns": 1.574600, "p95_decode_latency_ns": 2.047000},
    # 其余数据点...
]
data_pd = [
    {"N": 1000, "type": "PD", "avg_encode_latency_ns": 1.062400, "p95_encode_latency_ns": 1.295000, "avg_decode_latency_ns": 1.294700, "p95_decode_latency_ns": 1.402000},
    # 其余数据点...
]
data_mr = [
    {"N": 1000, "type": "MR", "avg_encode_latency_ns": 1.809700, "p95_encode_latency_ns": 2.523000, "avg_decode_latency_ns": 1.681200, "p95_decode_latency_ns": 2.372000},
    # 其余数据点...
]
data_cq = [
    {"N": 1000, "type": "CQ", "avg_encode_latency_ns": 2.148900, "p95_encode_latency_ns": 2.895000, "avg_decode_latency_ns": 2.045700, "p95_decode_latency_ns": 2.837000},
    # 其余数据点...
]

df_qp = pd.DataFrame(data_qp)
df_pd = pd.DataFrame(data_pd)
df_mr = pd.DataFrame(data_mr)
df_cq = pd.DataFrame(data_cq)

# 将所有数据集连接成一个DataFrame，并添加"type"列以区分来源
df_all = pd.concat([df_qp.assign(type='QP'), df_pd.assign(type='PD'), df_mr.assign(type='MR'), df_cq.assign(type='CQ')])

# 设置绘图风格
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 10})

# 图1: 平均延迟 vs N（4对象分面）
g = sns.FacetGrid(df_all, col="type", hue="type", sharey=False)
g.map(sns.lineplot, "N", "avg_encode_latency_ns", marker="o").add_legend()
g.map(sns.lineplot, "N", "avg_decode_latency_ns", marker="x").add_legend()
g.set(xscale="log")
g.set_axis_labels("N (log scale)", "Average Latency (ns)")
g.fig.suptitle('Average Encode/Decode Latency vs N')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("average_latency_vs_n.png", dpi=300)

# 图2: P95延迟 vs N（4对象分面）
g = sns.FacetGrid(df_all, col="type", hue="type", sharey=False)
g.map(sns.lineplot, "N", "p95_encode_latency_ns", marker="o").add_legend()
g.map(sns.lineplot, "N", "p95_decode_latency_ns", marker="x").add_legend()
g.set(xscale="log")
g.set_axis_labels("N (log scale)", "P95 Latency (ns)")
g.fig.suptitle('P95 Encode/Decode Latency vs N')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("p95_latency_vs_n.png", dpi=300)

# 图3: 固定N下跨对象对比（这里省略）

# 图4: 累积概率CDF
fig, ax = plt.subplots(figsize=(8, 5))
for type_name in ['QP', 'PD', 'MR', 'CQ']:
    df = df_all[df_all['type'] == type_name]
    for i, row in df.iterrows():
        N = row["N"]
        encode_mean = row["avg_encode_latency_ns"]
        decode_mean = row["avg_decode_latency_ns"]
        std = encode_mean * 0.1  # 假设标准差为平均值的10%

        # 生成样本并计算CDF
        samples = np.random.normal(encode_mean, std, 1000)
        sorted_samples = np.sort(samples)
        cdf = np.arange(1, len(sorted_samples)+1) / len(sorted_samples)
        ax.plot(sorted_samples, cdf, label=f'{type_name} Encode (N={N})')

        samples = np.random.normal(decode_mean, std, 1000)
        sorted_samples = np.sort(samples)
        cdf = np.arange(1, len(sorted_samples)+1) / len(sorted_samples)
        ax.plot(sorted_samples, cdf, label=f'{type_name} Decode (N={N})')

ax.set_xlabel('Latency (纳秒)',fontproperties=zh_font)
ax.set_ylabel('累积概率 (CDF)',fontproperties=zh_font)
ax.legend(loc='lower right')
ax.grid(True, which="both", ls="--", linewidth=0.5)
ax.set_title('Encode/Decode Latency Distribution (近似CDF)',fontproperties=zh_font)
plt.tight_layout()
plt.savefig("cdf_latency_distribution.png", dpi=300)

# 图5: 吞吐量与带宽 vs N（需要补充其他类型的数据）
# 示例中仅展示了QP类型的数据，请根据实际情况补充其他类型的数据。