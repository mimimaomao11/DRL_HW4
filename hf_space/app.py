"""
HW4 — DRL Agent Interactive Demo
Supports: DQN vs PPO on CartPole-v1
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import imageio
import gradio as gr
import gymnasium as gym
from stable_baselines3 import DQN, PPO

# ── Load models once at startup ───────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

def load_model(name):
    path = os.path.join(MODEL_DIR, name)
    if "dqn" in name.lower():
        return DQN.load(path)
    return PPO.load(path)

dqn_model = load_model("dqn_cartpole")
ppo_model = load_model("ppo_cartpole")

MODELS = {"DQN": dqn_model, "PPO": ppo_model}
COLORS = {"DQN": "#1A3A5C", "PPO": "#2E7DBF"}


# ── Run one episode, return frames + reward ───────────────
def run_episode(model, env):
    obs, _ = env.reset()
    frames, total_r = [], 0.0
    for _ in range(600):
        frames.append(env.render())
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, _ = env.step(action)
        total_r += r
        if terminated or truncated:
            break
    return frames, total_r


# ── Gradio callback ───────────────────────────────────────
def run_demo(algo_choice, n_episodes):
    model = MODELS[algo_choice]
    env   = gym.make("CartPole-v1", render_mode="rgb_array")

    all_frames, rewards = [], []
    for ep in range(int(n_episodes)):
        frames, r = run_episode(model, env)
        rewards.append(r)
        # Add episode label overlay
        for i, frame in enumerate(frames):
            img = frame.copy()
            all_frames.append(img)
        # Short black separator between episodes
        sep = np.zeros_like(frames[0])
        all_frames.extend([sep] * 6)

    env.close()

    # Save MP4
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    imageio.mimsave(tmp.name, all_frames, fps=30)

    # Stats text
    mean_r = np.mean(rewards)
    std_r  = np.std(rewards)
    stats  = (
        f"**{algo_choice}** — {n_episodes} episodes\n\n"
        f"| Episode | Reward |\n|---------|--------|\n"
        + "\n".join(f"| {i+1} | {r:.0f} |" for i, r in enumerate(rewards))
        + f"\n\n**Average: {mean_r:.1f} ± {std_r:.1f}**"
        + ("\n\n✅ CartPole solved!" if mean_r >= 475 else "")
    )
    return tmp.name, stats


def compare_both(n_episodes):
    results = {}
    for name, model in MODELS.items():
        env = gym.make("CartPole-v1", render_mode="rgb_array")
        rewards = []
        for _ in range(int(n_episodes)):
            _, r = run_episode(model, env)
            rewards.append(r)
        env.close()
        results[name] = rewards

    # Plot comparison
    fig = plt.figure(figsize=(10, 4))
    gs  = gridspec.GridSpec(1, 2, wspace=0.35)

    # Bar chart
    ax1 = fig.add_subplot(gs[0])
    algos = list(results.keys())
    means = [np.mean(results[a]) for a in algos]
    stds  = [np.std(results[a])  for a in algos]
    bars  = ax1.bar(algos, means, yerr=stds, capsize=8, width=0.45,
                    color=[COLORS[a] for a in algos], alpha=0.88)
    for bar, m in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width()/2, m + 5,
                 f"{m:.0f}", ha="center", fontsize=12, fontweight="bold")
    ax1.axhline(500, color="green", ls="--", lw=1.3, alpha=0.6, label="Max (500)")
    ax1.set(title="Average Reward Comparison", ylabel="Reward", ylim=(0, 560))
    ax1.legend(); ax1.grid(axis="y", alpha=0.3)
    ax1.spines[["top","right"]].set_visible(False)

    # Scatter per-episode
    ax2 = fig.add_subplot(gs[1])
    for name in algos:
        x = range(1, len(results[name]) + 1)
        ax2.plot(x, results[name], "o-", color=COLORS[name],
                 lw=2, label=name, markersize=7)
    ax2.axhline(500, color="green", ls="--", lw=1.3, alpha=0.6)
    ax2.set(title="Per-Episode Reward", xlabel="Episode", ylabel="Reward")
    ax2.legend(); ax2.grid(alpha=0.3)
    ax2.spines[["top","right"]].set_visible(False)

    fig.suptitle("DQN vs PPO — CartPole-v1 Live Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(tmp.name, dpi=130, bbox_inches="tight")
    plt.close()

    summary = "| Algorithm | Mean Reward | Std |\n|-----------|------------|-----|\n"
    for name in algos:
        summary += f"| **{name}** | {np.mean(results[name]):.1f} | {np.std(results[name]):.1f} |\n"
    return tmp.name, summary


# ── Gradio UI ─────────────────────────────────────────────
with gr.Blocks(
    title="DRL HW4 — Trained Agent Demo",
    theme=gr.themes.Soft(primary_hue="blue"),
    css=".gradio-container { max-width: 900px; margin: auto; }"
) as demo:

    gr.Markdown("""
    # DRL HW4 — Trained Agent Interactive Demo
    **DQN vs PPO on CartPole-v1** | Pre-trained models from Homework 4

    Watch trained reinforcement learning agents balance a pole in real-time!
    """)

    with gr.Tabs():

        # ── Tab 1: Single Agent Demo ──────────────────────
        with gr.Tab("Single Agent"):
            with gr.Row():
                algo  = gr.Radio(["DQN", "PPO"], value="PPO", label="Algorithm")
                n_ep  = gr.Slider(1, 5, value=2, step=1, label="Number of Episodes")
            run_btn   = gr.Button("Run Simulation", variant="primary")

            with gr.Row():
                video = gr.Video(label="Agent Playing CartPole", height=320)
                stats = gr.Markdown()

            run_btn.click(fn=run_demo, inputs=[algo, n_ep], outputs=[video, stats])

        # ── Tab 2: DQN vs PPO Comparison ─────────────────
        with gr.Tab("DQN vs PPO Comparison"):
            n_ep2    = gr.Slider(2, 10, value=5, step=1, label="Episodes per Algorithm")
            cmp_btn  = gr.Button("Run Comparison", variant="primary")

            with gr.Row():
                cmp_plot = gr.Image(label="Comparison Chart", height=300)
                cmp_text = gr.Markdown()

            cmp_btn.click(fn=compare_both, inputs=[n_ep2], outputs=[cmp_plot, cmp_text])

        # ── Tab 3: About ──────────────────────────────────
        with gr.Tab("About"):
            gr.Markdown("""
            ## Experiment Details

            | Item | Detail |
            |------|--------|
            | Environment | CartPole-v1 (OpenAI Gymnasium) |
            | Training steps | 60,000 |
            | DQN eval reward | 144.4 |
            | PPO eval reward | 500.0 ✅ |
            | Framework | Stable-Baselines3 2.8.0 |

            ## Algorithm Notes

            **DQN** (Deep Q-Network) — Off-policy, discrete actions, replay buffer.
            Peaks early (~episode 300) then suffers catastrophic forgetting.

            **PPO** (Proximal Policy Optimization) — On-policy, clipped surrogate objective.
            Stable convergence to perfect score. Industry standard for RLHF (ChatGPT, Claude).

            ## Source Code
            [GitHub Repository](https://github.com/mimimaomao11/DRL_HW4)
            """)

demo.launch()
