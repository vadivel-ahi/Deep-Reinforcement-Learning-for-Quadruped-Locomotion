# ===== ADD SYMMETRIC_FIXED TO COMPLETE RESULTS =====

from google.colab import drive
drive.mount('/content/drive')

import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

backup_folder = '/content/drive/MyDrive/RL_RewardShaping_20260411_1333'

# Load existing results
df_existing = pd.read_csv(f'{backup_folder}/final_results.csv')

print("="*70)
print("ADDING SYMMETRIC_FIXED TO RESULTS")
print("="*70)
print("\nExisting variants:")
print(df_existing['Variant'].tolist())

# Evaluate symmetric_fixed (quick - already trained!)
print("\nEvaluating symmetric_fixed...")

def symmetric_gentle_mod(obs, action, reward, info):
    quat = obs[3:7]
    if quat[0] > 0.9:
        ja = obs[7:15]
        left = np.concatenate([ja[0:2], ja[4:6]])
        right = np.concatenate([ja[2:4], ja[6:8]])
        asymmetry = np.sum(np.abs(left - right))
        return reward + 0.05 * np.exp(-asymmetry)
    else:
        return reward - 0.5

model = PPO.load(f'{backup_folder}/symmetric_fixed_FINAL')
env = gym.make('Ant-v5')

_step = env.step
env.step = lambda a: (lambda o,r,t,tr,i: (o, symmetric_gentle_mod(o,a,r,i), t, tr, i))(*_step(a))

rewards, lengths, controls = [], [], []

for ep in range(20):
    obs, _ = env.reset()
    r, l, c = 0, 0, 0
    
    for _ in range(1000):
        a, _ = model.predict(obs, deterministic=True)
        obs, rew, d, t, _ = env.step(a)
        r += rew
        l += 1
        c += np.sum(a**2)
        if d or t:
            break
    
    rewards.append(r)
    lengths.append(l)
    controls.append(c)

env.close()

print(f"  Mean Reward:  {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
print(f"  Mean Length:  {np.mean(lengths):.1f}")
print(f"  Mean Control: {np.mean(controls):.1f}")

# Add to results DataFrame
new_row = {
    'Variant': 'Symmetric_Fixed',
    'Mean_Reward': np.mean(rewards),
    'Std_Reward': np.std(rewards),
    'Mean_Length': np.mean(lengths),
    'Std_Length': np.std(lengths),
    'Mean_Control': np.mean(controls),
    'Mean_Velocity': 0  # Not tracked
}

# Combine with existing results
df_complete = pd.concat([df_existing, pd.DataFrame([new_row])], ignore_index=True)

print("\nUPDATED RESULTS TABLE:")
print(df_complete[['Variant', 'Mean_Reward', 'Mean_Length', 'Mean_Control']].to_string(index=False))

# Save updated results
df_complete.to_csv(f'{backup_folder}/complete_results_with_fixed.csv', index=False)

# CREATE UPDATED COMPARISON PLOT (5 variants)
print("\nCreating updated comparison plot...")

fig, axes = plt.subplots(2, 2, figsize=(16, 11))

colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']  # Added purple for fixed
x = np.arange(len(df_complete))

# Plot 1: Rewards
ax1 = axes[0, 0]
bars = ax1.bar(x, df_complete['Mean_Reward'], yerr=df_complete['Std_Reward'],
               capsize=7, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
ax1.set_xticks(x)
ax1.set_xticklabels(df_complete['Variant'], fontsize=11, rotation=20, ha='right')
ax1.set_ylabel('Mean Episode Reward', fontsize=13, fontweight='bold')
ax1.set_title('A) Reward Performance', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
ax1.axhline(y=0, color='black', linewidth=1)

for i, bar in enumerate(bars):
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., h + df_complete['Std_Reward'].iloc[i] + 50,
             f'{h:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 2: Episode Length
ax2 = axes[0, 1]
bars2 = ax2.bar(x, df_complete['Mean_Length'], yerr=df_complete['Std_Length'],
                capsize=5, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
ax2.set_xticks(x)
ax2.set_xticklabels(df_complete['Variant'], fontsize=11, rotation=20, ha='right')
ax2.set_ylabel('Episode Length', fontsize=13, fontweight='bold')
ax2.set_title('B) Stability', fontsize=14, fontweight='bold')
ax2.axhline(y=1000, color='red', linestyle='--', linewidth=2, label='Max', alpha=0.7)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Control Cost
ax3 = axes[1, 0]
bars3 = ax3.bar(x, df_complete['Mean_Control'], color=colors, alpha=0.85,
                edgecolor='black', linewidth=2)
ax3.set_xticks(x)
ax3.set_xticklabels(df_complete['Variant'], fontsize=11, rotation=20, ha='right')
ax3.set_ylabel('Control Cost', fontsize=13, fontweight='bold')
ax3.set_title('C) Energy Efficiency', fontsize=14, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

for bar in bars3:
    h = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., h + 10,
             f'{h:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 4: Summary Table
ax4 = axes[1, 1]
ax4.axis('off')

table_data = []
for _, row in df_complete.iterrows():
    table_data.append([
        row['Variant'],
        f"{row['Mean_Reward']:.0f}",
        f"{row['Mean_Length']:.0f}",
        f"{row['Mean_Control']:.0f}"
    ])

table = ax4.table(cellText=table_data,
                  colLabels=['Variant', 'Reward', 'Length', 'Energy'],
                  cellLoc='center', loc='center',
                  colWidths=[0.35, 0.22, 0.22, 0.22])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.2)

# Header styling
for i in range(4):
    table[(0, i)].set_facecolor('#2c3e50')
    table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)

# Row colors
for i, color in enumerate(colors, 1):
    for j in range(4):
        table[(i, j)].set_facecolor(f'{color}25')

fig.suptitle('Reward Shaping Comparison (Including Symmetric_Fixed)',
             fontsize=17, fontweight='bold', y=0.98)

plt.tight_layout()

plt.savefig('complete_comparison_5variants.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(f'{backup_folder}/complete_comparison_5variants.png', dpi=300, bbox_inches='tight', facecolor='white')

plt.show()

print("\n✅ Updated comparison plot created with 5 variants!")

# Download
from google.colab import files
files.download('complete_comparison_5variants.png')
files.download(f'{backup_folder}/complete_results_with_fixed.csv')

print("\nCOMPLETE!")
print("\nYour final 5 variants:")
for _, row in df_complete.iterrows():
    print(f"  • {row['Variant']:<17}: {row['Mean_Reward']:>6.0f} reward")
