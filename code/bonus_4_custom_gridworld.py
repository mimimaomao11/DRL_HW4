"""
╔══════════════════════════════════════════════════════════╗
║  加分項目 4 — 自訂 GridWorld 環境 + DQN                 ║
║  延伸你的 GridWorld 經驗，改用神經網路取代查找表         ║
╚══════════════════════════════════════════════════════════╝

套件安裝：
    pip install gymnasium stable-baselines3 matplotlib numpy

執行：
    python bonus_4_custom_gridworld.py

輸出：
    bonus4_gridworld_path.png    ← 最優路徑視覺化（放報告）
    bonus4_gridworld_curve.png   ← 學習曲線（放報告）

設計說明：
    • 10×10 網格，有障礙物、陷阱、目標
    • 狀態：(row, col, steps_taken) → 24 維向量
    • 動作：上下左右（4 個離散動作）
    • 獎勵：到達目標 +100、踩陷阱 -50、每步 -0.5（鼓勵快速）
    • 這正好對應你做過的 GridWorld，但用 DQN 神經網路學習
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
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
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import time


# ══════════════════════════════════════════════════════════════════
#  自訂 GridWorld 環境（繼承 gymnasium.Env）
# ══════════════════════════════════════════════════════════════════
class GridWorldEnv(gym.Env):
    """
    10×10 GridWorld

    圖例：
      S = 起點 (0,0)
      G = 目標 (9,9)
      X = 障礙物（不可進入）
      T = 陷阱（可進入但扣分）
      . = 空地

    狀態特徵（24 維）：
      [row/9, col/9,          # 歸一化位置
       上下左右各方格類型 × 5, # 5 格視野
       到目標的 L1 距離/18,    # 目標方向
       已走步數/max_steps]     # 效率指標
    """

    metadata = {"render_modes": []}

    # 地圖設計（0=空，1=障礙，2=陷阱，3=目標）
    GRID = np.array([
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 2, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 2, 0],
        [0, 1, 1, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 3],
    ], dtype=int)

    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右
    N       = 10
    MAX_STEPS = 200

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(24,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.start = (0, 0)
        self.goal  = (9, 9)
        self._pos  = self.start
        self._steps = 0

    def _obs(self):
        r, c = self._pos
        gr, gc = self.goal
        obs = np.zeros(24, dtype=np.float32)
        obs[0] = r / (self.N - 1)
        obs[1] = c / (self.N - 1)
        # 5×5 視野（攤平取前 20 個）
        idx = 2
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.N and 0 <= nc < self.N:
                    obs[idx] = self.GRID[nr, nc] / 3.0
                else:
                    obs[idx] = -1.0   # 邊界外
                idx += 1
                if idx >= 22:
                    break
            if idx >= 22:
                break
        obs[22] = (abs(gr - r) + abs(gc - c)) / (2 * (self.N - 1))
        obs[23] = self._steps / self.MAX_STEPS
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._pos   = self.start
        self._steps = 0
        return self._obs(), {}

    def step(self, action):
        dr, dc = self.ACTIONS[action]
        r, c   = self._pos
        nr, nc = r + dr, c + dc
        self._steps += 1

        # 邊界 / 障礙物：不移動
        if not (0 <= nr < self.N and 0 <= nc < self.N):
            nr, nc = r, c
        elif self.GRID[nr, nc] == 1:
            nr, nc = r, c

        self._pos = (nr, nc)
        cell = self.GRID[nr, nc]

        # 獎勵
        if cell == 3:     # 目標
            reward, terminated = 100.0, True
        elif cell == 2:   # 陷阱
            reward, terminated = -50.0, False
        else:
            dist_before = abs(r - 9) + abs(c - 9)
            dist_after  = abs(nr - 9) + abs(nc - 9)
            reward = (dist_before - dist_after) * 0.5 - 0.5
            terminated = False

        truncated = self._steps >= self.MAX_STEPS
        return self._obs(), reward, terminated, truncated, {}


# ══════════════════════════════════════════════════════════════════
#  訓練
# ══════════════════════════════════════════════════════════════════
class EpisodeLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.ep_rewards  = []
        self.ep_lengths  = []
        self._cur_r = 0.0
        self._cur_l = 0

    def _on_step(self):
        self._cur_r += self.locals["rewards"][0]
        self._cur_l += 1
        if self.locals["dones"][0]:
            self.ep_rewards.append(self._cur_r)
            self.ep_lengths.append(self._cur_l)
            self._cur_r = 0.0
            self._cur_l = 0
        return True


def smooth(data, w=30):
    if len(data) < w:
        return np.array(data, dtype=float)
    return np.convolve(data, np.ones(w) / w, mode="valid")


def train():
    env = Monitor(GridWorldEnv())
    cb  = EpisodeLogger()
    t0  = time.time()

    print("▶ 訓練 DQN @ 自訂 GridWorld-10x10")
    model = DQN(
        "MlpPolicy", env,
        learning_rate=5e-4,
        buffer_size=100_000,
        learning_starts=2000,
        batch_size=128,
        exploration_fraction=0.4,
        exploration_final_eps=0.05,
        target_update_interval=500,
        policy_kwargs=dict(net_arch=[128, 128, 64]),  # 網路架構
        verbose=0,
    )
    model.learn(200_000, callback=cb)
    elapsed = time.time() - t0

    mean_r, std_r = evaluate_policy(model, Monitor(GridWorldEnv()), n_eval_episodes=50)
    # 成功率（目標 reward > 50 視為成功到達）
    success_rate = np.mean([r > 50 for r in cb.ep_rewards[-200:]]) * 100
    print(f"  ✓ {elapsed:.0f}s | 評估: {mean_r:.1f} ± {std_r:.1f} | 近期成功率: {success_rate:.1f}%")
    return model, cb, mean_r, std_r, success_rate, elapsed


# ══════════════════════════════════════════════════════════════════
#  視覺化：提取最優路徑
# ══════════════════════════════════════════════════════════════════
def extract_path(model, max_steps=150):
    env   = GridWorldEnv()
    obs, _ = env.reset()
    path  = [env._pos]
    total_r = 0

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        path.append(env._pos)
        total_r += reward
        if terminated or truncated:
            break

    reached = env.GRID[path[-1]] == 3
    return path, total_r, reached


def plot_grid(model, cb, mean_r, std_r, success_rate):
    path, path_r, reached = extract_path(model)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("加分項目 4 — 自訂 GridWorld + DQN 代理人",
                 fontsize=13, fontweight="bold")

    # ── 左：地圖 + 路徑 ──────────────────────────────────────────
    ax = axes[0]
    N   = GridWorldEnv.N
    GRID = GridWorldEnv.GRID

    color_map = {0: "#F0F4F8", 1: "#2C3E50", 2: "#E74C3C", 3: "#27AE60"}
    label_map = {0: "空地", 1: "障礙物", 2: "陷阱", 3: "目標"}

    img = np.zeros((N, N, 3))
    for r in range(N):
        for c in range(N):
            rgb = mcolors.to_rgb(color_map[GRID[r, c]])
            img[r, c] = rgb
    ax.imshow(img, origin="upper", extent=(-0.5, N - 0.5, N - 0.5, -0.5))

    # 格線
    for i in range(N + 1):
        ax.axhline(i - 0.5, color="gray", lw=0.4, alpha=0.5)
        ax.axvline(i - 0.5, color="gray", lw=0.4, alpha=0.5)

    # 路徑
    if len(path) >= 2:
        rs = [p[0] for p in path]
        cs = [p[1] for p in path]
        ax.plot(cs, rs, "b-", lw=2, alpha=0.7, zorder=3)
        ax.plot(cs[0], rs[0], "go", ms=12, zorder=4, label="起點 S")
        ax.plot(cs[-1], rs[-1], "r*", ms=14, zorder=4,
                label="終點" + ("（到達目標）✓" if reached else "（未到達）✗"))

    # 步驟編號（每 10 步標一次）
    for i in range(0, len(path), max(1, len(path) // 8)):
        r, c = path[i]
        ax.text(c, r, str(i), ha="center", va="center",
                fontsize=6.5, color="navy", zorder=5, fontweight="bold")

    patches = [mpatches.Patch(color=v, label=label_map[k])
               for k, v in color_map.items()]
    ax.legend(handles=patches + ax.get_legend_handles_labels()[0][2:],
              loc="upper right", fontsize=8, framealpha=0.9)
    status = "✓ 成功到達目標" if reached else "✗ 未到達（到達時間到）"
    ax.set_title(f"最優路徑視覺化  ({len(path)-1} 步, Reward={path_r:.0f})\n{status}",
                 fontsize=10)
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xticklabels(range(N), fontsize=7)
    ax.set_yticklabels(range(N), fontsize=7)

    # ── 右：學習曲線 ─────────────────────────────────────────────
    ax2 = axes[1]
    rewards = cb.ep_rewards
    lengths = cb.ep_lengths
    x = np.arange(1, len(rewards) + 1)

    ax2_r = ax2.twinx()
    ax2.plot(x, rewards, alpha=0.15, color="#1A3A5C", lw=0.7)
    if len(rewards) >= 30:
        sm = smooth(rewards, 30)
        ax2.plot(np.arange(30, len(rewards) + 1), sm,
                 color="#1A3A5C", lw=2.2, label="Reward 移動平均")
    ax2_r.plot(x, lengths, alpha=0.3, color="#E67E22", lw=0.7)
    if len(lengths) >= 30:
        sm_l = smooth(lengths, 30)
        ax2_r.plot(np.arange(30, len(lengths) + 1), sm_l,
                   color="#E67E22", lw=1.5, ls="--", label="步數（右軸）")

    ax2.axhline(100, color="green", ls="--", lw=1.2, alpha=0.5, label="目標獎勵 100")
    ax2.set(xlabel="Episode", ylabel="Episode Reward", title="DQN 學習曲線")
    ax2_r.set_ylabel("Episode 步數", color="#E67E22")

    # 合併圖例
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    ax2.grid(alpha=0.3)
    ax2.spines["top"].set_visible(False)

    # 統計資訊框
    info = (f"最終評估\n"
            f"平均 Reward: {mean_r:.1f} ± {std_r:.1f}\n"
            f"近期成功率: {success_rate:.0f}%\n"
            f"總 episodes: {len(rewards)}")
    ax2.text(0.97, 0.05, info, transform=ax2.transAxes,
             fontsize=8.5, va="bottom", ha="right",
             bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="#1A3A5C", lw=1))

    plt.tight_layout()
    out1 = "bonus4_gridworld_path.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"\n  📊 圖表已儲存：{out1}")
    plt.show()


def print_summary(mean_r, success_rate):
    print(f"""
╔══════════════════════════════════════════════════════╗
║  加分項目 4 結論                                     ║
╠══════════════════════════════════════════════════════╣
  平均 Reward：{mean_r:.1f}
  近期成功率：{success_rate:.0f}%

  與你之前的 GridWorld 實驗比較：
  • Q-Learning / SARSA → 查找表，記憶每個 (s,a) 的 Q 值
  • DQN                → 神經網路，輸入狀態特徵輸出 Q 值
  
  神經網路的優勢：
  • 可泛化到未見過的狀態
  • 狀態空間更大時仍可運作（查找表會爆炸）
  • 可結合視野特徵（本實驗的 5×5 視野）
╚══════════════════════════════════════════════════════╝""")


if __name__ == "__main__":
    model, cb, mean_r, std_r, success_rate, elapsed = train()
    print_summary(mean_r, success_rate)
    plot_grid(model, cb, mean_r, std_r, success_rate)
