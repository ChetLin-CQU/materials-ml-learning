import pandas as pd
import matplotlib.pyplot as plt
data = {
    "合金": ["AZ31", "AZ61", "AZ91"],
    "铝含量_wt%": [3.0, 6.0, 9.0],
    "抗拉强度_MPa": [260, 280, 230],
}

df = pd.DataFrame(data)

print("材料数据表：")
print(df)

average_strength = df["抗拉强度_MPa"].mean()
print(f"\n平均抗拉强度：{average_strength:.1f} MPa")

plt.figure(figsize=(7, 5))

plt.plot(
    df["铝含量_wt%"],
    df["抗拉强度_MPa"],
    marker="o",
    linewidth=2,
)

plt.xlabel("Al Content (wt%)")
plt.ylabel("Tensile Strength (MPa)")
plt.title("Al Content vs Tensile Strength")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("alloy_strength.png", dpi=300)
plt.show()