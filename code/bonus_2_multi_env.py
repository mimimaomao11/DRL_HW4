"""
╔══════════════════════════════════════════════════════════╗
║  加分項目 2 — PPO 多環境比較                             ║
║  CartPole / MountainCar / LunarLander 三種難度           ║
╚══════════════════════════════════════════════════════════╝

套件安裝：
    pip install gymnasium stable-baselines3 matplotlib numpy
    pip install gymnasium[box2d]   # LunarLander 需要此套件

執行：
    python bonus_2_multi_env.py

輸出：
    bonus2_multi_env.png  ← 放入報告
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

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


# ── 各環境的訓練設定 ──────────────────────────────────────────────
ENV_CONFIGS = {
    "CartPole-v1": {
        "steps":     80_000,
        "color":     "#1A3A5C",
        "success":   475,        # 連續 100 episodes 平均達到視為解決
        "ylim":      (0, 560),
        "desc":      "離散動作 / 簡單 / 快速收斂",
        "ppo_kw":    dict(n_steps=512, batch_size=64, n_epochs=10,
                         learning_rate=3e-4, clip_range=0.2, ent_coef=0.01),
    },
    "MountainCar-v0": {
        "steps":     200_000,
        "color":     "#2E7DBF",
        "success":   -110,       # 越靠近 0 越好
        "ylim":      (-210, 0),
        "desc":      "稀疏獎勵 / 需探索 / 較難收斂",
        "ppo_kw":    dict(n_steps=2048, batch_size=64, n_epochs=10,
                         learning_rate=3e-4, clip_range=0.2, ent_coef=0.05),
    },
    # LunarLander 需要 Box2D；若安裝失敗會自動改用 Acrobot-v1
    "LunarLander-v3": {
        "steps":     300_000,
        "color":     "#0A6E9F",
        "success":   200,
        "ylim":      (-400, 300),
        "desc":      "離散動作 / 複雜物理 / 最難（需 Box2D）",
        "ppo_kw":    dict(n_steps=1024, batch_size=64, n_epochs=4,
                         learning_rate=3e-4, clip_range=0.2, ent_coef=0.01),
    },
    # ↓ Box2D 安裝失敗時的備用環境（不需任何額外套件）
    "Acrobot-v1": {
        "steps":     200_000,
        "color":     "#0A6E9F",
        "success":   -100,
        "ylim":      (-510, -50),
        "desc":      "離散動作 / 稀疏獎勵 / 中高難度（備用）",
        "ppo_kw":    dict(n_steps=1024, batch_size=64, n_epochs=10,
                         learning_rate=3e-4, clip_range=0.2, ent_coef=0.01),
    },
}


def train_env(env_id, cfg):
    print(f"\n▶ 訓練 PPO @ {env_id}  ({cfg['steps']:,} steps)")
    env = Monitor(gym.make(env_id))
    cb  = EpisodeLogger()
    t0  = time.time()

    model = PPO("MlpPolicy", env, verbose=0, **cfg["ppo_kw"])
    model.learn(cfg["steps"], callback=cb)

    elapsed = time.time() - t0
    mean_r, std_r = evaluate_policy(
        model, Monitor(gym.make(env_id)), n_eval_episodes=20)
    env.close()

    print(f"  ✓ {elapsed:.0f}s | 評估: {mean_r:.1f} ± {std_r:.1f} | episodes: {len(cb.ep_rewards)}")
    return cb.ep_rewards, mean_r, std_r, elapsed


def plot(results):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    fig.suptitle("加分項目 2 — PPO 多環境比較", fontsize=13, fontweight="bold")

    for ax, (env_id, (rewards, mean_r, std_r, elapsed)) in zip(axes, results.items()):
        cfg = ENV_CONFIGS[env_id]
        x   = np.arange(1, len(rewards) + 1)
        w   = 20

        ax.plot(x, rewards, alpha=0.2, color=cfg["color"], lw=0.8)
        if len(rewards) >= w:
            sm = smooth(rewards, w)
            ax.plot(np.arange(w, len(rewards) + 1), sm,
                    color=cfg["color"], lw=2.2, label="移動平均")

        ax.axhline(cfg["success"], color="green", ls="--", lw=1.3, alpha=0.7,
                   label=f"解決門檻 {cfg['success']}")

        ax.set_title(f"{env_id.split('-')[0]}\n{cfg['desc']}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Episode"); ax.set_ylabel("Reward")
        ax.set_ylim(cfg["ylim"])
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # 右上角資訊框
        info = f"最終: {mean_r:.0f} ± {std_r:.0f}\n訓練: {elapsed:.0f}s"
        ax.text(0.97, 0.97, info, transform=ax.transAxes,
                fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=cfg["color"], lw=1))

    plt.tight_layout()
    out = "bonus2_multi_env.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  📊 圖表已儲存：{out}")
    plt.show()


def print_summary(results):
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  多環境比較結論                                      ║")
    print("╠══════════════════════════════════════════════════════╣")
    for env_id, (_, mean_r, std_r, elapsed) in results.items():
        cfg = ENV_CONFIGS[env_id]
        solved = "✓ 解決" if (
            (cfg["success"] > 0 and mean_r >= cfg["success"]) or
            (cfg["success"] < 0 and mean_r >= cfg["success"])
        ) else "✗ 未完全解決"
        print(f"  {env_id:<20} {mean_r:7.1f} ± {std_r:5.1f}  {elapsed:5.0f}s  {solved}")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  結論：任務複雜度越高，需要更多訓練步數；              ║")
    print("║  MountainCar 稀疏獎勵最難，需仔細調整探索係數。       ║")
    print("╚══════════════════════════════════════════════════════╝")


def select_envs():
    """自動偵測 Box2D 是否可用，選擇對應的環境組合"""
    selected = ["CartPole-v1", "MountainCar-v0"]
    try:
        import Box2D  # noqa
        selected.append("LunarLander-v3")
        print("✓ Box2D 可用 → 使用 LunarLander-v3")
    except ImportError:
        selected.append("Acrobot-v1")
        print("⚠ Box2D 未安裝 → 改用 Acrobot-v1（備用環境）")
        print("  如需 LunarLander：pip install swig && pip install \"gymnasium[box2d]\"")
    return {k: ENV_CONFIGS[k] for k in selected}


if __name__ == "__main__":
    envs = select_envs()
    results = {}
    for env_id, cfg in envs.items():
        rewards, mean_r, std_r, elapsed = train_env(env_id, cfg)
        results[env_id] = (rewards, mean_r, std_r, elapsed)

    print_summary(results)
    plot(results)
