"""
========================================================
 作業四 加分項目 — DQN vs PPO CartPole-v1 訓練實驗
========================================================
目標：
  1. 訓練 DQN 代理人（你已熟悉的演算法）
  2. 訓練 PPO 代理人（業界 RLHF 主力）
  3. 比較兩者的學習曲線
  4. 生成可放入報告的圖表

環境需求：
  pip install gymnasium stable-baselines3 matplotlib numpy

執行方式：
  python main.py
========================================================
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import time

# ─────────────────────────────────────────────────────────
# 設定中文字體（Windows：微軟正黑體 / macOS：蘋方 / Linux：Noto Sans CJK）
# ─────────────────────────────────────────────────────────
def _setup_chinese_font():
    """自動偵測並設定可顯示中文的字體"""
    candidates = [
        "Microsoft JhengHei",   # Windows 繁體
        "Microsoft YaHei",      # Windows 簡體
        "PingFang TC",          # macOS 繁體
        "PingFang SC",          # macOS 簡體
        "Noto Sans CJK TC",     # Linux 繁體
        "Noto Sans CJK SC",     # Linux 簡體
        "WenQuanYi Micro Hei",  # Linux 備用
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            plt.rcParams["axes.unicode_minus"] = False  # 修正負號顯示
            return font
    # 找不到中文字體時，退回英文標題（避免亂碼）
    return None

_CHINESE_FONT = _setup_chinese_font()


# ─────────────────────────────────────────────────────────
# Callback：每個 episode 記錄 reward
# ─────────────────────────────────────────────────────────
class RewardLogger(BaseCallback):
    """記錄每個 episode 的 reward，方便後續繪圖"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self._current_episode_reward = 0.0

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        done = self.locals["dones"][0]
        self._current_episode_reward += reward
        if done:
            self.episode_rewards.append(self._current_episode_reward)
            self._current_episode_reward = 0.0
        return True


# ─────────────────────────────────────────────────────────
# 訓練函式
# ─────────────────────────────────────────────────────────
def train_agent(algo_name, total_timesteps=60000):
    """
    訓練指定演算法的代理人

    參數:
        algo_name: "DQN" 或 "PPO"
        total_timesteps: 訓練總步數（步數越多，學習越充分）

    回傳:
        model: 訓練好的模型
        callback: 包含 episode_rewards 的 callback 物件
        elapsed: 訓練花費秒數
    """
    env = gym.make("CartPole-v1")
    callback = RewardLogger()

    print(f"\n{'='*50}")
    print(f"  開始訓練 {algo_name} — CartPole-v1")
    print(f"  總步數：{total_timesteps:,}")
    print(f"{'='*50}")

    start = time.time()

    if algo_name == "DQN":
        # DQN 超參數說明：
        #   learning_rate     — 神經網路更新步長
        #   buffer_size       — 重播緩衝區大小（越大越穩定）
        #   learning_starts   — 開始訓練前先收集這麼多步資料
        #   batch_size        — 每次從緩衝區取多少筆資料更新
        #   exploration_fraction — ε 從 1.0 衰減到最小值所佔的訓練比例
        model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=1e-3,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=64,
            exploration_fraction=0.3,
            exploration_final_eps=0.02,
            train_freq=4,
            target_update_interval=500,
            verbose=0,
        )

    elif algo_name == "PPO":
        # PPO 超參數說明：
        #   n_steps       — 每次收集多少步再更新（rollout buffer 大小）
        #   batch_size    — mini-batch 大小
        #   n_epochs      — 每次 rollout 資料重複使用幾次
        #   clip_range    — 截斷比例（PPO 的核心，防止策略更新過大）
        #   ent_coef      — 熵獎勵係數（鼓勵探索）
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=0,
        )

    model.learn(total_timesteps=total_timesteps, callback=callback)
    elapsed = time.time() - start

    print(f"  ✅ 訓練完成！耗時 {elapsed:.1f} 秒")
    print(f"  📊 共完成 {len(callback.episode_rewards)} 個 episodes")

    env.close()
    return model, callback, elapsed


# ─────────────────────────────────────────────────────────
# 平滑函式（移動平均）
# ─────────────────────────────────────────────────────────
def smooth(data, window=20):
    """計算移動平均，讓曲線更易讀"""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


# ─────────────────────────────────────────────────────────
# 評估代理人
# ─────────────────────────────────────────────────────────
def evaluate(model, n_eval=20):
    """評估訓練好的模型，回傳平均 reward 與標準差"""
    env = gym.make("CartPole-v1")
    mean_r, std_r = evaluate_policy(model, env, n_eval_episodes=n_eval)
    env.close()
    return mean_r, std_r


# ─────────────────────────────────────────────────────────
# 繪製比較圖（放入報告用）
# ─────────────────────────────────────────────────────────
def plot_comparison(dqn_rewards, ppo_rewards, dqn_eval, ppo_eval):
    """
    生成三張子圖：
      (1) DQN 學習曲線
      (2) PPO 學習曲線
      (3) 最終平均 Reward 柱狀比較
    """
    # 根據是否有中文字體，決定標題語言
    if _CHINESE_FONT:
        main_title = "DQN vs PPO — CartPole-v1 學習曲線比較\n(作業四 加分項目)"
        legend_smooth = "移動平均（window=20）"
        legend_perfect = "完美分數 (500)"
        title_suffix = "學習曲線"
        bar_title = "最終評估（20 episodes）"
        bar_ylabel = "平均 Reward ± std"
        best_label_prefix = "最高"
    else:
        main_title = "DQN vs PPO — CartPole-v1 Learning Curve Comparison\n(Bonus Assignment)"
        legend_smooth = "Moving Avg (window=20)"
        legend_perfect = "Perfect Score (500)"
        title_suffix = "Learning Curve"
        bar_title = "Final Eval (20 episodes)"
        bar_ylabel = "Mean Reward ± std"
        best_label_prefix = "Best"

    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(main_title, fontsize=14, fontweight="bold", y=1.01)

    gs = gridspec.GridSpec(1, 3, width_ratios=[2, 2, 1.2], wspace=0.35)

    colors = {"DQN": "#1A3A5C", "PPO": "#2E7DBF"}

    for idx, (name, rewards) in enumerate([("DQN", dqn_rewards), ("PPO", ppo_rewards)]):
        ax = fig.add_subplot(gs[idx])
        episodes = range(1, len(rewards) + 1)
        # 原始 reward（淡色）
        ax.plot(episodes, rewards, alpha=0.25, color=colors[name], linewidth=0.8)
        # 移動平均（主要曲線）
        sm = smooth(rewards, window=20)
        ax.plot(range(20, len(rewards) + 1), sm, color=colors[name], linewidth=2.0,
                label=legend_smooth)
        # 最高分標記
        best_ep = np.argmax(rewards) + 1
        best_r = max(rewards)
        ax.scatter(best_ep, best_r, color="orange", s=60, zorder=5)
        ax.annotate(f"{best_label_prefix}: {best_r:.0f}", xy=(best_ep, best_r),
                    xytext=(10, -15), textcoords="offset points", fontsize=8, color="darkorange")
        # CartPole 成功線（500 = 完美）
        ax.axhline(y=500, color="green", linestyle="--", linewidth=1.2, alpha=0.6, label=legend_perfect)
        ax.set_title(f"{name} {title_suffix}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Episode", fontsize=10)
        ax.set_ylabel("Episode Reward", fontsize=10)
        ax.set_ylim(0, 550)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # 柱狀比較圖
    ax3 = fig.add_subplot(gs[2])
    algos = ["DQN", "PPO"]
    means = [dqn_eval[0], ppo_eval[0]]
    stds  = [dqn_eval[1], ppo_eval[1]]
    bars = ax3.bar(algos, means, yerr=stds, capsize=8, width=0.5,
                   color=[colors["DQN"], colors["PPO"]],
                   edgecolor="white", linewidth=1.5, alpha=0.9)
    # 數值標籤
    for bar, mean, std in zip(bars, means, stds):
        ax3.text(bar.get_x() + bar.get_width()/2, mean + std + 10,
                 f"{mean:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax3.axhline(y=500, color="green", linestyle="--", linewidth=1.2, alpha=0.6)
    ax3.set_title(bar_title, fontsize=11, fontweight="bold")
    ax3.set_ylabel(bar_ylabel, fontsize=10)
    ax3.set_ylim(0, 580)
    ax3.grid(True, axis="y", alpha=0.3)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    plt.tight_layout()
    filename = "drl_cartpole_comparison.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"\n  📈 圖表已儲存：{filename}（請放入作業報告）")
    plt.show()


# ─────────────────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  作業四 加分項目 — DQN vs PPO CartPole 實驗")
    print("  你的背景：GridWorld / DQN / Q-Learning / SARSA")
    print("  新挑戰：PPO（業界 RLHF 主力演算法）")
    print("=" * 55)

    TIMESTEPS = 60000  # 可調高到 100000 獲得更好的結果

    # 訓練 DQN
    dqn_model, dqn_cb, dqn_time = train_agent("DQN", TIMESTEPS)

    # 訓練 PPO
    ppo_model, ppo_cb, ppo_time = train_agent("PPO", TIMESTEPS)

    # 評估
    print("\n" + "=" * 50)
    print("  評估訓練結果（各跑 20 個 episodes）")
    print("=" * 50)
    dqn_eval = evaluate(dqn_model)
    ppo_eval = evaluate(ppo_model)
    print(f"\n  DQN  — 平均 Reward: {dqn_eval[0]:.1f} ± {dqn_eval[1]:.1f}  （訓練耗時 {dqn_time:.1f}s）")
    print(f"  PPO  — 平均 Reward: {ppo_eval[0]:.1f} ± {ppo_eval[1]:.1f}  （訓練耗時 {ppo_time:.1f}s）")
    print(f"\n  CartPole-v1 最高分為 500（完美平衡）")

    # 比較分析（加入報告的文字）
    print("\n" + "=" * 50)
    print("  比較分析（可直接放入報告）")
    print("=" * 50)
    winner = "PPO" if ppo_eval[0] >= dqn_eval[0] else "DQN"
    print(f"""
  DQN（Deep Q-Network）採用 off-policy Q-learning，
  透過重播緩衝區（Replay Buffer）與目標網路穩定訓練，
  與你在 GridWorld 實驗中使用的 Q-learning 原理相同，
  但以神經網路取代查找表，處理連續狀態空間。

  PPO（Proximal Policy Optimization）採用 on-policy 策略
  梯度法，透過截斷代理目標函數（Clipped Surrogate Objective）
  限制每次更新幅度，避免策略崩潰。PPO 是目前 RLHF
  （ChatGPT、Claude 對齊訓練）與機器人控制的業界標準。

  實驗結果：{winner} 在本次實驗中表現較佳。
  兩者差異可能因隨機種子與超參數設定而有所不同。
    """)

    # 繪圖
    plot_comparison(dqn_cb.episode_rewards, ppo_cb.episode_rewards, dqn_eval, ppo_eval)

    # 儲存模型（選用）
    dqn_model.save("dqn_cartpole")
    ppo_model.save("ppo_cartpole")
    print("  💾 模型已儲存：dqn_cartpole.zip / ppo_cartpole.zip")
    print("\n✅  加分項目完成！請將 drl_cartpole_comparison.png 放入報告。")


if __name__ == "__main__":
    main()