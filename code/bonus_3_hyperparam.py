"""
╔══════════════════════════════════════════════════════════╗
║  加分項目 3 — PPO 超參數消融實驗                         ║
║  系統性比較 learning_rate × clip_range 對訓練的影響      ║
╚══════════════════════════════════════════════════════════╝

套件安裝：
    pip install gymnasium stable-baselines3 matplotlib numpy seaborn

執行：
    python bonus_3_hyperparam.py

輸出：
    bonus3_hyperparam_heatmap.png   ← 熱力圖（放報告）
    bonus3_hyperparam_curves.png    ← 學習曲線（放報告）

注意：此實驗跑 12 種組合 × 每組 3 seeds，約需 15–25 分鐘。
      可調整 SEEDS = [0] 來加速（只跑 1 個 seed）。
"""

import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors

# ── 設定中文字體 ─────────────────────────────────────────────────
def _setup_chinese_font():
    candidates = [
        "Microsoft JhengHei", "Microsoft YaHei",
        "PingFang TC", "PingFang SC",
        "Noto Sans CJK TC", "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            plt.rcParams["axes.unicode_minus"] = False
            return font
    return None

_setup_chinese_font()

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# ── 實驗設定 ──────────────────────────────────────────────────────
LR_LIST    = [1e-4, 3e-4, 1e-3]          # 學習率
CLIP_LIST  = [0.1, 0.2, 0.3, 0.4]        # 截斷比例
SEEDS      = [0, 1, 2]                    # 隨機種子（改成 [0] 可加速）
STEPS      = 60_000                       # 每組訓練步數
N_EVAL     = 20                           # 每組評估次數


class EpisodeLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.ep_rewards = []
        self._cur = 0.0

    def _on_step(self):
        self._cur += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.ep_rewards.append(self._cur)
            self._cur = 0.0
        return True


def smooth(data, w=20):
    if len(data) < w:
        return np.array(data, dtype=float)
    return np.convolve(data, np.ones(w) / w, mode="valid")


def run_one(lr, clip, seed):
    env = Monitor(gym.make("CartPole-v1"))
    cb  = EpisodeLogger()
    model = PPO(
        "MlpPolicy", env,
        learning_rate=lr,
        clip_range=clip,
        n_steps=512, batch_size=64, n_epochs=10,
        ent_coef=0.01, seed=seed, verbose=0,
    )
    model.learn(STEPS, callback=cb)
    mean_r, std_r = evaluate_policy(model, Monitor(gym.make("CartPole-v1")), n_eval_episodes=N_EVAL)
    env.close()
    return mean_r, std_r, cb.ep_rewards


# ── 執行所有組合 ──────────────────────────────────────────────────
def run_ablation():
    total = len(LR_LIST) * len(CLIP_LIST) * len(SEEDS)
    print(f"▶ 共 {total} 個實驗 ({len(LR_LIST)} lr × {len(CLIP_LIST)} clip × {len(SEEDS)} seeds)")
    print(f"  每組 {STEPS:,} steps，預計 {total * 0.8:.0f}–{total * 1.5:.0f} 分鐘\n")

    # results[lr][clip] = list of (mean_r, seed)
    results   = {lr: {clip: [] for clip in CLIP_LIST} for lr in LR_LIST}
    curves    = {lr: {clip: [] for clip in CLIP_LIST} for lr in LR_LIST}
    done = 0

    for lr, clip, seed in itertools.product(LR_LIST, CLIP_LIST, SEEDS):
        t0 = time.time()
        mean_r, std_r, ep_r = run_one(lr, clip, seed)
        elapsed = time.time() - t0
        results[lr][clip].append(mean_r)
        curves[lr][clip].append(ep_r)
        done += 1
        print(f"  [{done:2d}/{total}] lr={lr:.0e}  clip={clip:.1f}  seed={seed}"
              f"  →  {mean_r:6.1f} ± {std_r:.1f}  ({elapsed:.0f}s)")

    return results, curves


# ── 熱力圖 ────────────────────────────────────────────────────────
def plot_heatmap(results):
    matrix = np.array([
        [np.mean(results[lr][clip]) for clip in CLIP_LIST]
        for lr in LR_LIST
    ])

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=500, aspect="auto")
    plt.colorbar(im, ax=ax, label="平均 Reward（CartPole-v1 最高 500）")

    ax.set_xticks(range(len(CLIP_LIST)))
    ax.set_yticks(range(len(LR_LIST)))
    ax.set_xticklabels([f"clip={c}" for c in CLIP_LIST])
    ax.set_yticklabels([f"lr={lr:.0e}" for lr in LR_LIST])
    ax.set_xlabel("Clip Range (PPO 截斷比例)")
    ax.set_ylabel("Learning Rate (學習率)")
    ax.set_title("加分項目 3 — PPO 超參數消融實驗\n（熱力圖：顏色越深 = Reward 越高）",
                 fontweight="bold")

    # 標數字
    for i in range(len(LR_LIST)):
        for j in range(len(CLIP_LIST)):
            val = matrix[i, j]
            color = "white" if val > 350 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)

    # 找最佳組合
    best_i, best_j = np.unravel_index(np.argmax(matrix), matrix.shape)
    ax.add_patch(plt.Rectangle(
        (best_j - 0.5, best_i - 0.5), 1, 1,
        fill=False, edgecolor="orange", lw=3, label="最佳組合"
    ))
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    out = "bonus3_hyperparam_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  📊 熱力圖已儲存：{out}")

    best_lr   = LR_LIST[best_i]
    best_clip = CLIP_LIST[best_j]
    print(f"  🏆 最佳組合：lr={best_lr:.0e}  clip={best_clip}  "
          f"Reward={matrix[best_i, best_j]:.0f}")
    return best_lr, best_clip


# ── 學習曲線（比較不同 lr，固定最佳 clip）────────────────────────
def plot_curves(curves, best_clip):
    fig, axes = plt.subplots(1, len(LR_LIST), figsize=(5 * len(LR_LIST), 4))
    fig.suptitle(f"加分項目 3 — 不同 Learning Rate 學習曲線\n(clip_range={best_clip}，3 seeds 平均)",
                 fontsize=12, fontweight="bold")

    palette = ["#1A3A5C", "#2E7DBF", "#0A6E9F"]

    for ax, (lr, color) in zip(axes, zip(LR_LIST, palette)):
        seed_curves = curves[lr][best_clip]
        # 對齊長度（取最短）
        min_len = min(len(c) for c in seed_curves)
        aligned = np.array([c[:min_len] for c in seed_curves])
        mean_c  = aligned.mean(axis=0)
        std_c   = aligned.std(axis=0)

        x = np.arange(1, min_len + 1)
        ax.fill_between(x, mean_c - std_c, mean_c + std_c,
                        alpha=0.2, color=color)
        ax.plot(x, mean_c, color=color, lw=2, label=f"lr={lr:.0e}")

        if min_len >= 20:
            sm = smooth(mean_c)
            ax.plot(np.arange(20, min_len + 1), sm,
                    color=color, lw=2.5, ls="--", label="平滑")

        ax.axhline(500, color="green", ls="--", lw=1.2, alpha=0.6)
        ax.set(title=f"lr = {lr:.0e}", xlabel="Episode",
               ylabel="Reward", ylim=(0, 560))
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = "bonus3_hyperparam_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  📊 學習曲線已儲存：{out}")
    plt.show()


def print_summary(results):
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  超參數消融實驗結論                                  ║")
    print("╠══════════════════════════════════════════════════════╣")
    all_means = [(lr, clip, np.mean(results[lr][clip]))
                 for lr in LR_LIST for clip in CLIP_LIST]
    all_means.sort(key=lambda x: -x[2])
    print("  排名  lr          clip   平均 Reward")
    for rank, (lr, clip, m) in enumerate(all_means[:5], 1):
        print(f"  #{rank}    {lr:.0e}    {clip:.1f}    {m:.1f}")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  結論：learning rate 過大易不穩定；                   ║")
    print("║  clip_range 過小收斂慢，過大策略更新不受控。          ║")
    print("╚══════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    results, curves = run_ablation()
    print_summary(results)
    best_lr, best_clip = plot_heatmap(results)
    plot_curves(curves, best_clip)
