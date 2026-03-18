import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

base_dir = Path("/data/nas/users/ruijie.sang/logs/wmnav_qwen_train1/ObjectNav_wmnav-Qwen2_5-VL-7B-Instruct-hm3dv2")
save_dir = Path("/data/nas/users/ruijie.sang/logs") # 保存在当前运行目录下

results = []

for instance_dir in sorted(base_dir.iterdir()):
    if not instance_dir.is_dir() or "_of_" not in instance_dir.name:
        continue
    for episode_dir in sorted(instance_dir.iterdir()):
        if not episode_dir.is_dir():
            continue
        pkl_path = episode_dir / "df_results.pkl"
        if not pkl_path.exists():
            continue
        try:
            df = pd.read_pickle(pkl_path)
            last_row = df.iloc[-1]
            results.append({
                "instance": instance_dir.name,
                "episode": episode_dir.name,
                "object": str(last_row.get("object", "unknown")),
                "success": int(last_row["finish_status"] == "success"),
                "spl": float(last_row["spl"]),
                "goal_reached": bool(last_row["goal_reached"]),
                "distance_to_goal": float(last_row["distance_to_goal"]),
                "finish_status": str(last_row["finish_status"]),
                "num_steps": len(df),
            })
        except Exception as e:
            print(f"Error reading {pkl_path}: {e}")

df_all = pd.DataFrame(results)

print("=" * 60)
print(f"Total episodes found: {len(df_all)}")
print(f"Expected: 50 × 20 = 1000")
print("=" * 60)

# 核心指标
sr = df_all["success"].mean()
spl = df_all["spl"].mean()
avg_dist = df_all["distance_to_goal"].mean()
avg_steps = df_all["num_steps"].mean()

df_succ = df_all[df_all["success"] == 1]
succ_avg_dist = df_succ["distance_to_goal"].mean() if len(df_succ) > 0 else 0
succ_avg_steps = df_succ["num_steps"].mean() if len(df_succ) > 0 else 0

print(f"\n{'Metric':<30} {'Value':>10}")
print("-" * 45)
print(f"{'Success Rate (SR)':<30} {sr:>10.4f}")
print(f"{'SPL':<30} {spl:>10.4f}")
print(f"{'Avg Distance to Goal':<30} {avg_dist:>10.4f}")
print(f"{'Avg Steps':<30} {avg_steps:>10.1f}")
print(f"{'Succ Avg Distance to Goal':<30} {succ_avg_dist:>10.4f}")
print(f"{'Succ Avg Steps':<30} {succ_avg_steps:>10.1f}")

# finish_status 分布
print(f"\nFinish Status Distribution:")
status_counts = Counter(df_all["finish_status"])
for status, count in status_counts.most_common():
    print(f"  {status:<20} {count:>5}  ({count/len(df_all)*100:.1f}%)")

# 按物品类别统计成功率
print(f"\nPer-Object Success Rate:")
obj_stats = df_all.groupby("object").agg(
    total=("success", "count"),
    successes=("success", "sum"),
    sr=("success", "mean"),
    avg_spl=("spl", "mean"),
).sort_values("sr", ascending=False)
for obj, row in obj_stats.iterrows():
    print(f"  {obj:<20} SR={row['sr']:.3f}  SPL={row['avg_spl']:.3f}  ({int(row['successes'])}/{int(row['total'])})")

# 保存完整结果
df_all.to_csv(save_dir / "summary_results.csv", index=False)
print(f"\nSaved: {save_dir / 'summary_results.csv'}")

# 保存失败任务详情
df_fail = df_all[df_all["success"] == 0].copy()
df_fail = df_fail.sort_values(["finish_status", "instance", "episode"])
df_fail.to_csv(save_dir / "failed_episodes.csv", index=False)
print(f"Saved: {save_dir / 'failed_episodes.csv'}  ({len(df_fail)} failed episodes)")