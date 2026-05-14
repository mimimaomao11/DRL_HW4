"""
╔══════════════════════════════════════════════════════════╗
║  加分項目 5 — SAC LunarLander 連續控制                  ║
║  最大熵框架 × 連續動作空間 × 著陸影片錄製               ║
╚══════════════════════════════════════════════════════════╝

套件安裝：
    pip install gymnasium stable-baselines3 matplotlib numpy
    pip install gymnasium[box2d]     # LunarLander 需要
    pip install imageio imageio-ffmpeg  # 錄製影片需要

執行：
    python bonus_5_sac_lunarlander.py

輸出：
    bonus5_sac_curve.png      ← 學習曲線（放報告）
    bonus5_landing.gif        ← 著陸動畫（放報告 / 簡報）
    bonus5_entropy.png        ← SAC 熵值變化（展示最大熵特性）

說明：
    LunarLanderContinuous-v3 需要輸出連續推力 [-1, 1]²
    DQN 無法處理（離散動作），SAC 的最大熵框架專為此設計。
    這正是 SAC 比 DQN/PPO 更適合機器人控制的原因。
"""

import time
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import torch as th

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
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


# ── Callback：記錄 reward + 熵值 ─────────────────────────────────
class SACLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.ep_rewards   = []
        self.entropy_log  = []  # SAC 的熵值（每 1000 步記一次）
        self._cur_r = 0.0
        self._step  = 0

    def _on_step(self):
        self._cur_r += self.locals["rewards"][0]
        self._step  += 1
        if self.locals["dones"][0]:
            self.ep_rewards.append(self._cur_r)
            self._cur_r = 0.0
        # 每 1000 步記錄一次 entropy 係數
        if self._step % 1000 == 0:
            try:
                if hasattr(self.model, "log_ent_coef"):
                    ent = float(th.exp(self.model.log_ent_coef).item())
                else:
                    ent = float(self.model.ent_coef_tensor.item())
                self.entropy_log.append(ent)
            except Exception:
                pass
        return True


def smooth(data, w=20):
    if len(data) < w:
        return np.array(data, dtype=float)
    return np.convolve(data, np.ones(w) / w, mode="valid")


# ── 訓練 ──────────────────────────────────────────────────────────
def train():
    env = Monitor(gym.make("LunarLanderContinuous-v3"))
    cb  = SACLogger()
    t0  = time.time()

    print(">> 訓練 SAC @ LunarLanderContinuous-v3")
    print("  (連續動作空間：主引擎推力 + 側向推力，各 [-1, 1])")

    model = SAC(
        "MlpPolicy", env,
        learning_rate=3e-4,
        buffer_size=100_000,   # 縮小以降低記憶體與採樣成本
        learning_starts=2000,  # 更早開始學習
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=2,          # 每 2 步更新一次（降低 CPU 負擔）
        gradient_steps=1,
        ent_coef="auto",
        target_entropy="auto",
        policy_kwargs=dict(net_arch=[128, 128]),  # 縮小網路加速推理
        verbose=0,
    )

    model.learn(150_000, callback=cb)  # 150k 步：~15-25 分鐘，仍可達 >200
    elapsed = time.time() - t0

    mean_r, std_r = evaluate_policy(model, Monitor(gym.make("LunarLanderContinuous-v3")), n_eval_episodes=10)
    env.close()

    print(f"  [OK] {elapsed:.0f}s | 評估: {mean_r:.1f} +/- {std_r:.1f} | episodes: {len(cb.ep_rewards)}")
    return model, cb, mean_r, std_r, elapsed


# ── 錄製 GIF ─────────────────────────────────────────────────────
def record_gif(model, filename="bonus5_landing.gif", n_episodes=3):
    try:
        import imageio
    except ImportError:
        print("  [WARN]imageio 未安裝，跳過 GIF 錄製。安裝：pip install imageio imageio-ffmpeg")
        return False

    env    = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")
    frames = []
    total_r = 0

    print(f"\n>> 錄製 {n_episodes} 個 episode -> {filename}")
    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_r   = 0
        for _ in range(1000):
            frame = env.render()
            frames.append(frame)
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            ep_r += r
            if terminated or truncated:
                break
        total_r += ep_r
        print(f"  Episode {ep+1}: Reward = {ep_r:.0f}")

    env.close()
    imageio.mimsave(filename, frames, fps=30, loop=0)
    print(f"  [OK]GIF 已儲存：{filename}  ({len(frames)} frames)")
    return True


# ── 繪圖 ──────────────────────────────────────────────────────────
def plot(cb, mean_r, std_r):
    fig = plt.figure(figsize=(14, 5))
    fig.suptitle("加分項目 5 — SAC LunarLanderContinuous-v3",
                 fontsize=13, fontweight="bold")
    gs  = gridspec.GridSpec(1, 3, width_ratios=[4, 4, 3], wspace=0.35)

    # ── 學習曲線 ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    rewards = cb.ep_rewards
    x = np.arange(1, len(rewards) + 1)
    ax1.plot(x, rewards, alpha=0.2, color="#1A3A5C", lw=0.8)
    if len(rewards) >= 30:
        sm = smooth(rewards, 30)
        ax1.plot(np.arange(30, len(rewards)+1), sm,
                 color="#1A3A5C", lw=2.2, label="移動平均")
    ax1.axhline(200, color="green", ls="--", lw=1.3, alpha=0.6,
                label="成功著陸門檻 200")
    ax1.axhline(0, color="gray", ls=":", lw=1.0, alpha=0.5)
    ax1.set(title="SAC 學習曲線", xlabel="Episode",
            ylabel="Episode Reward", ylim=(-500, 350))
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # 資訊框
    info = f"最終評估\n{mean_r:.1f} ± {std_r:.1f}"
    ax1.text(0.97, 0.05, info, transform=ax1.transAxes,
             fontsize=9, va="bottom", ha="right",
             bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="#1A3A5C"))

    # ── 熵值變化（SAC 最大熵特性）────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    if cb.entropy_log:
        ent_x = np.arange(1, len(cb.entropy_log) + 1) * 1000
        ax2.plot(ent_x, cb.entropy_log, color="#E67E22", lw=2.2)
        ax2.fill_between(ent_x, 0, cb.entropy_log, alpha=0.2, color="#E67E22")
        ax2.set(title="SAC 自動熵調整（ent_coef）",
                xlabel="Training Steps", ylabel="Entropy Coefficient α")
        ax2.grid(alpha=0.3)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.text(0.05, 0.95,
                 "α 越大 → 探索更多\nα 越小 → 利用更多\nSAC 自動平衡",
                 transform=ax2.transAxes, fontsize=9, va="top",
                 bbox=dict(boxstyle="round", fc="lightyellow", ec="#E67E22"))
    else:
        ax2.text(0.5, 0.5, "熵值資料不足\n（需更多訓練步數）",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=11)
        ax2.set_title("SAC 自動熵調整")

    # ── SAC 核心原理說明 ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.axis("off")
    text = (
        "SAC 核心原理\n"
        "─────────────────────\n"
        "目標函數：\n"
        "max π [ Σ E[r + α·H(π)] ]\n\n"
        "H(π) = 策略的熵（隨機性）\n"
        "α    = 自動調整的溫度係數\n\n"
        "優點：\n"
        "• 自動平衡探索 vs 利用\n"
        "• 連續動作空間 SOTA\n"
        "• Off-policy → 樣本效率高\n"
        "• 不需手動調 ε（探索率）\n\n"
        "vs DQN：\n"
        "  DQN 只能離散動作\n"
        "  SAC 輸出連續推力\n\n"
        "vs PPO：\n"
        "  PPO on-policy，樣本少\n"
        "  SAC off-policy，更高效"
    )
    ax3.text(0.05, 0.95, text, transform=ax3.transAxes,
             fontsize=9, va="top",
             bbox=dict(boxstyle="round,pad=0.6", fc="#F0F4F8", ec="#1A3A5C", lw=1.2))

    plt.tight_layout()
    out = "bonus5_sac_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  [PNG]學習曲線已儲存：{out}")
    plt.show()


def print_summary(mean_r, elapsed):
    solved = "[OK] 成功解決（> 200）" if mean_r >= 200 else f"訓練中（{mean_r:.0f}，可增加步數）"
    print(f"""
====================================================
  加分項目 5 結論
====================================================
  任務：LunarLanderContinuous-v3（連續動作）
  演算法：SAC（Soft Actor-Critic, 最大熵框架）
  訓練時間：{elapsed:.0f}s
  最終評估：{mean_r:.1f}  {solved}

  為什麼選 SAC 而不是 DQN？
  - LunarLander 的動作是連續值（推力大小）
  - DQN 只能輸出離散動作（無法使用）
  - SAC 透過最大熵目標同時最大化 reward + 策略隨機性
  - 自動熵調整（ent_coef='auto'）無需手動設定探索率

  機器人應用關聯：
  - Isaac Gym / Unitree 機器人訓練均採用 SAC/TD3
  - 連續關節力矩控制完全對應此任務設定
====================================================""")


if __name__ == "__main__":
    model, cb, mean_r, std_r, elapsed = train()
    print_summary(mean_r, elapsed)
    plot(cb, mean_r, std_r)
    record_gif(model)  # 需要安裝 imageio，若無則自動跳過
