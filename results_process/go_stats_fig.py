import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
import os

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 9,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'axes.unicode_minus': False
})


# ========= 1. 读取稀疏矩阵 =========
# 假设文件在当前工作目录下
path = '/l/users/ding.bai/scFoundation/pert/data/go_mask_19264.npz'
#path = "go_mask_19264.npz"
assert os.path.exists(path), f"{path} not found"

# 这是一个 N x N 的邻接矩阵，z[i, j] = 1 表示 i 和 j 至少共享一个 GO term
go_mask = load_npz(path)
N = go_mask.shape[0]
print(f"Loaded GO mask of shape {go_mask.shape}")

# ========= 2. 计算每个 gene 的邻居数量 =========
# 每一行的非零个数 = 该 gene 连接到多少其他 gene
row_degrees = np.asarray(go_mask.getnnz(axis=1), dtype=np.int64)

# 如果对角线上是 1（self-loop），可以减掉自己这一条连接
# 如果你不确定，可以看一下对角的最大值
diag = go_mask.diagonal()
print("Diagonal max/min:", diag.max(), diag.min())

# 如果对角线都是 1，可以考虑去掉 self：
row_degrees_no_self = row_degrees - 1
row_degrees_no_self = np.clip(row_degrees_no_self, 0, None)
# 下面这行根据你是否想去掉 self 来切换：
row_degrees_effective = row_degrees  # 或者改成 row_degrees_no_self

# ========= 3. 打印一些基本统计信息 =========
mean_deg = row_degrees_effective.mean()
median_deg = np.median(row_degrees_effective)
max_deg = row_degrees_effective.max()
min_deg = row_degrees_effective.min()

density = mean_deg / (N - 1)  # 平均每个 gene 连接到多少比例的其他基因

print(f"Mean neighbors per gene: {mean_deg:.2f}")
print(f"Median neighbors per gene: {median_deg}")
print(f"Min/Max neighbors: {min_deg} / {max_deg}")
print(f"Approx. average density per row: {density*100:.2f}%")

fig, ax = plt.subplots()
# ========= 5. 可选：log-scale 的 y 轴（更清楚尾部） =========
fig.set_size_inches(6, 4)
ax.hist(row_degrees_effective, bins=50)

'''
# Define bins for counting occurrences
counts, bin_edges = np.histogram(row_degrees_effective, bins=20)

# Compute bin centers
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

ax.plot(bin_centers, counts, marker=None, linestyle='-', color='black', label="Frequency")'''
ax.set_xlabel("Count of GO-overlapping genes per gene")
ax.set_ylabel("Number of genes")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.plot([1563, 1563], [0, 1100], marker=None, linestyle='--', color='red', label='red line')
ax.text(1563, 1100 + 100, f'Mean: {int(1563)}', color='red', ha='center', fontdict={'fontsize': 9})

ax.plot([max_deg, max_deg], [0, 300], marker=None, linestyle='--', color='red', label='red line')
ax.text(max_deg, 400, f'Max: {int(max_deg)}', color='red', ha='center', fontdict={'fontsize': 9})

fig.tight_layout()
fig.savefig("fig/go_mask_degree_dist.svg", format="svg", dpi=300)
fig.savefig("fig/go_mask_degree_dist.pdf", format="pdf", dpi=300)
plt.close()

'''
fig, ax = plt.subplots()
# ========= 5. 可选：log-scale 的 y 轴（更清楚尾部） =========
fig.set_size_inches(6, 4)
ax.hist(row_degrees_effective, bins=50)
ax.set_xlabel("Count of GO-overlapping genes per gene")
ax.set_ylabel("Number of genes (log scale)")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.plot([1563, 1563], [0, 2e3], marker=None, linestyle='--', color='red', label='Red Line')
ax.text(1563 + 400, 2e3 * 0.7, f'{int(1563)}', color='red', ha='center', fontdict={'fontsize': 9})

ax.set_yscale("log")

fig.tight_layout()
fig.savefig("fig/go_mask_degree_dist_log_scale.svg", format="svg", dpi=300)
fig.savefig("fig/go_mask_degree_dist_log_scale.pdf", format="pdf", dpi=300)


# ========= 6. 可选：按比例画直方图（邻居数量 / (N-1)） =========
row_density = row_degrees_effective / (N - 1)

plt.figure(figsize=(6, 4))
plt.hist(row_density, bins=50)
plt.xlabel("Fraction of genes connected via GO")
plt.ylabel("Number of genes")
#plt.title("GO mask row-wise density distribution")
plt.tight_layout()
plt.savefig("fig/go_mask_degree_dist_row_wise.svg", format="svg", dpi=300)
plt.savefig("fig/go_mask_degree_dist_row_wise.pdf", format="pdf", dpi=300)'''
