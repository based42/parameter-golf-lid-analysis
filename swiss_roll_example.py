import matplotlib.pyplot as plt
from sklearn import datasets

sr_points, sr_color = datasets.make_swiss_roll(n_samples=1500, random_state=0)

fig = plt.figure(figsize=(12, 5), constrained_layout=True)
fig.get_layout_engine().set(wspace=0.12)
ax_3d = fig.add_subplot(1, 2, 1, projection="3d")
scatter_3d = ax_3d.scatter(
    sr_points[:, 0], sr_points[:, 1], sr_points[:, 2], c=sr_color, s=30, alpha=0.9, cmap="viridis")
ax_3d.set_title(r"Ambient space ($D=3$)")
ax_3d.set_xlabel("$x$")
ax_3d.set_ylabel("$y$")
ax_3d.set_zlabel("$z$")
ax_3d.set_box_aspect(
    (1, 1, 0.8),
    zoom=1.2,
)
ax_3d.view_init(azim=-66, elev=12)
ax_3d.set_box_aspect((1, 1, 0.8), zoom=1.2)

# 2D intrinsic-coordinate representation
ax_2d = fig.add_subplot(1, 2, 2)
ax_2d.scatter(sr_color, sr_points[:, 1], c=sr_color, s=30, alpha=0.9, cmap="viridis")
ax_2d.set_title(r"Unrolled manifold ($d=2$)")
ax_2d.set_xlabel(r"Position in the roll $t$")
ax_2d.set_ylabel(r"Height $h$")

fig.colorbar(scatter_3d, ax=[ax_3d, ax_2d], label=r"Position in the roll $t$", shrink=0.8)

fig.savefig(
    "swiss_roll.pdf",
    bbox_inches="tight",
    dpi=300,
)

plt.show()