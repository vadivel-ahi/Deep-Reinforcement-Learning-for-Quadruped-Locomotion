# ===== REAL LEARNING CURVES FROM CHECKPOINTS =====

print("Installing packages...")
!pip install gymnasium[mujoco] stable-baselines3[extra] imageio -q > /dev/null 2>&1

from google.colab import drive
drive.mount('/content/drive')

import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

backup_folder = '/content/drive/MyDrive/RL_RewardShaping_20260411_1333'

print("="*70)
print("GENERATING REAL LEARNING CURVES FROM CHECKPOINTS")
print("="*70)
print("Estimated time: 10-15 minutes")
print("Evaluating 40 checkpoints (10 per variant × 4 variants)")
print("="*70 + "\n")

def evaluate_checkpoint(model_path, reward_modifier=None, num_episodes=5):
    """Evaluate a single checkpoint"""
    
    if not os.path.exists(f"{model_path}.zip"):
        return None
    
    try:
        model = PPO.load(model_path)
        env = gym.make('Ant-v5')
        
        if reward_modifier:
            _step = env.step
            env.step = lambda a: (lambda o,r,t,tr,i: (o, reward_modifier(o,a,r,i), t, tr, i))(*_step(a))
        
        rewards = []
        lengths = []
        
        for _ in range(num_episodes):
            obs, _ = env.reset()
            ep_reward = 0
            ep_length = 0
            
            for _ in range(1000):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = env.step(action)
                ep_reward += reward
                ep_length += 1
                if done or truncated:
                    break
            
            rewards.append(ep_reward)
            lengths.append(ep_length)
        
        env.close()
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_length': np.mean(lengths)
        }
    
    except Exception as e:
        print(f"      ❌ Error: {e}")
        return None

# Reward modifiers (same as training)
sym = lambda o,a,r,i: r + 0.05*np.exp(-np.sum(np.abs(np.concatenate([o[7:9],o[11:13]])-np.concatenate([o[9:11],o[13:15]]))))
eff = lambda o,a,r,i: r - 0.3*np.sum(a**2)
spd = lambda o,a,r,i: 1.3*i.get('x_velocity',0) - 0.4*np.sum(a**2) + 1.0

variants = {
    'baseline': None,
    'symmetric': sym,
    'efficient': eff,
    'speed': spd
}

# Collect learning curve data
learning_data = []

overall_start = time.time()

for variant_name, reward_mod in variants.items():
    print(f"\n{'='*70}")
    print(f"EVALUATING: {variant_name.upper()}")
    print(f"{'='*70}")
    
    variant_start = time.time()
    
    for checkpoint_num in range(1, 11):  # Checkpoints 1-10
        timesteps = checkpoint_num * 100_000
        checkpoint_path = f"{backup_folder}/{variant_name}_ckpt{checkpoint_num}"
        
        print(f"  Checkpoint {checkpoint_num:2d} ({timesteps:>7,} steps)...", end=' ', flush=True)
        
        result = evaluate_checkpoint(checkpoint_path, reward_mod, num_episodes=5)
        
        if result:
            learning_data.append({
                'Variant': variant_name.capitalize(),
                'Timesteps': timesteps,
                'Mean_Reward': result['mean_reward'],
                'Std_Reward': result['std_reward'],
                'Mean_Length': result['mean_length']
            })
            print(f"✅ Reward: {result['mean_reward']:>7.1f} ± {result['std_reward']:>5.1f}")
        else:
            print("❌ Failed")
    
    variant_time = (time.time() - variant_start) / 60
    print(f"\n  ⏱️  {variant_name} complete in {variant_time:.1f} minutes")

total_time = (time.time() - overall_start) / 60
print(f"\n{'='*70}")
print(f"✅ ALL CHECKPOINTS EVALUATED!")
print(f"Total time: {total_time:.1f} minutes")
print(f"{'='*70}")

# Create DataFrame
df_learning = pd.DataFrame(learning_data)

# Save data
df_learning.to_csv('learning_curves_data.csv', index=False)
df_learning.to_csv(f'{backup_folder}/learning_curves_data.csv', index=False)

print("\nLearning curve data saved!")

# Create the learning curve plot
print("\nCreating learning curve visualization...")

plt.figure(figsize=(14, 8))

colors = {
    'Baseline': '#3498db',
    'Symmetric': '#e74c3c',
    'Efficient': '#2ecc71',
    'Speed': '#f39c12'
}

markers = {
    'Baseline': 'o',
    'Symmetric': 's',
    'Efficient': '^',
    'Speed': 'D'
}

# Plot each variant
for variant in ['Baseline', 'Symmetric', 'Efficient', 'Speed']:
    variant_data = df_learning[df_learning['Variant'] == variant]
    
    if len(variant_data) > 0:
        timesteps = variant_data['Timesteps'].values
        rewards = variant_data['Mean_Reward'].values
        std_rewards = variant_data['Std_Reward'].values
        
        # Plot line with error bars
        plt.plot(timesteps, rewards,
                marker=markers[variant],
                linewidth=3,
                markersize=10,
                color=colors[variant],
                label=variant,
                alpha=0.9)
        
        # Add shaded error region
        plt.fill_between(timesteps,
                        rewards - std_rewards,
                        rewards + std_rewards,
                        color=colors[variant],
                        alpha=0.15)

# Formatting
plt.xlabel('Training Timesteps', fontsize=15, fontweight='bold')
plt.ylabel('Mean Episode Reward', fontsize=15, fontweight='bold')
plt.title('Learning Curves: Training Progression Across Reward Variants',
          fontsize=17, fontweight='bold', pad=20)

plt.legend(fontsize=13, loc='lower right', framealpha=0.95, 
          edgecolor='black', fancybox=True)

plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
plt.axhline(y=0, color='black', linewidth=1, alpha=0.5, linestyle='-')

# Format axes
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

# Set x-axis to show nice intervals
plt.xticks([0, 200_000, 400_000, 600_000, 800_000, 1_000_000],
          ['0', '200K', '400K', '600K', '800K', '1M'],
          fontsize=12)

plt.yticks(fontsize=12)

# Add subtle background
ax.set_facecolor('#f8f9fa')

plt.tight_layout()

# Save in multiple locations
plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(f'{backup_folder}/learning_curves.png', dpi=300, bbox_inches='tight', facecolor='white')

plt.show()

print("\n" + "="*70)
print("✅ LEARNING CURVES COMPLETE!")
print("="*70)

# Print summary
print("\nFinal Performance at 1M steps:")
for variant in ['Baseline', 'Symmetric', 'Efficient', 'Speed']:
    final_data = df_learning[(df_learning['Variant'] == variant) & 
                            (df_learning['Timesteps'] == 1_000_000)]
    if len(final_data) > 0:
        final_reward = final_data['Mean_Reward'].values[0]
        final_std = final_data['Std_Reward'].values[0]
        print(f"  {variant:<12}: {final_reward:>7.1f} ± {final_std:>5.1f}")

print("\nDownloading learning curve plot...")
from google.colab import files
files.download('learning_curves.png')
files.download('learning_curves_data.csv')

print("\nALL DONE! You now have:")
print("   ✅ Learning curve plot (line graph)")
print("   ✅ Learning curve data (CSV)")
print("   ✅ Comparison plot (bar graphs)")
print("   ✅ 4 gait videos")
print("   ✅ Results table")
print("\nREADY TO CREATE POSTER!")
print("="*70)
