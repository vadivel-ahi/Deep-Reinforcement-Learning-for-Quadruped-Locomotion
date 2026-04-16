# ===== SYMMETRIC FIXED - NO FLIPPING ALLOWED =====

!pip install gymnasium[mujoco] stable-baselines3[extra] imageio -q

from google.colab import drive
drive.mount('/content/drive')

import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import time
import imageio

backup_folder = '/content/drive/MyDrive/RL_RewardShaping_20260411_1333'

print("="*70)
print(" TRAINING: SYMMETRIC_FIXED (Orientation-Aware)")
print("="*70)
print(" Modification: Symmetry bonus ONLY when upright")
print("  Heavy penalty for flipping upside-down")
print(" Time: ~30 minutes")
print("="*70 + "\n")

def symmetric_fixed_modifier(obs, action, reward, info):
    """
    Symmetric gait bonus with orientation constraint
    
    Key difference from original:
    - Checks if ant is upright using quaternion
    - Only gives symmetry bonus when upright
    - Heavily penalizes flipping
    """
    # Get orientation from quaternion (obs[3:7])
    quat = obs[3:7]  # [w, x, y, z]
    
    # Upright check: w component should be close to 1
    # w > 0.9 means ant is right-side up
    # w < 0.9 means ant is tilted or inverted
    is_upright = quat[0] > 0.9
    
    if is_upright:
        # ANT IS UPRIGHT - give symmetry bonus
        
        # Extract joint angles
        joint_angles = obs[7:15]
        
        # Left legs (front-left + back-left)
        left_legs = np.concatenate([
            joint_angles[0:2],   # Front-left: hip, ankle
            joint_angles[4:6]    # Back-left: hip, ankle
        ])
        
        # Right legs (front-right + back-right)
        right_legs = np.concatenate([
            joint_angles[2:4],   # Front-right: hip, ankle
            joint_angles[6:8]    # Back-right: hip, ankle
        ])
        
        # Compute asymmetry
        asymmetry = np.sum(np.abs(left_legs - right_legs))
        
        # Symmetry bonus (exponential decay)
        symmetry_bonus = 0.05 * np.exp(-asymmetry)
        
        return reward + symmetry_bonus
    
    else:
        # ANT IS UPSIDE DOWN 
        flip_penalty = 0.5
        return reward - flip_penalty

# Create environment with fixed modifier
env = gym.make('Ant-v5')

_original_step = env.step

def modified_step(action):
    obs, reward, terminated, truncated, info = _original_step(action)
    modified_reward = symmetric_fixed_modifier(obs, action, reward, info)
    return obs, modified_reward, terminated, truncated, info

env.step = modified_step

# Create model
model = PPO("MlpPolicy", env, verbose=0, device='cpu')

print(" Starting training (1,000,000 steps)...\n")
start_time = time.time()

# Train in 10 chunks with progress updates
for i in range(10):
    chunk_start = time.time()
    print(f"  Chunk {i+1}/10 ({(i+1)*10}%)...", end=' ', flush=True)
    
    model.learn(
        total_timesteps=100_000,
        reset_num_timesteps=False,
        progress_bar=False
    )
    
    # Save checkpoint to Drive
    model.save(f"{backup_folder}/symmetric_fixed_ckpt{i+1}")
    
    chunk_time = (time.time() - chunk_start) / 60
    print(f"✅ ({chunk_time:.1f}min)")

# Save final model to both locations
model.save("symmetric_fixed_FINAL")
model.save(f"{backup_folder}/symmetric_fixed_FINAL")

print(f"\n Saved to Drive: {backup_folder}/symmetric_fixed_FINAL.zip ✅")

# Evaluation
print("\n Running evaluation (20 episodes)...")

env_eval = gym.make('Ant-v5')
_step_eval = env_eval.step
env_eval.step = lambda a: (lambda o,r,t,tr,i: (o, symmetric_fixed_modifier(o,a,r,i), t, tr, i))(*_step_eval(a))

eval_rewards = []
eval_lengths = []
upright_count = 0

for ep in range(20):
    obs, _ = env_eval.reset()
    ep_reward = 0
    ep_length = 0
    was_upright = True
    
    for step in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env_eval.step(action)
        
        # Check if stayed upright
        if obs[3] < 0.9:  # Flipped!
            was_upright = False
        
        ep_reward += reward
        ep_length += 1
        
        if done or truncated:
            break
    
    eval_rewards.append(ep_reward)
    eval_lengths.append(ep_length)
    if was_upright:
        upright_count += 1
    
    if ep < 5:
        orientation_status = "✅ Upright" if was_upright else "❌ Flipped"
        print(f"  Ep {ep+1}: {ep_reward:7.1f} reward, {ep_length:4d} steps - {orientation_status}")

print(f"\n Evaluation Summary:")
print(f"  Mean Reward:  {np.mean(eval_rewards):7.1f} ± {np.std(eval_rewards):5.1f}")
print(f"  Mean Length:  {np.mean(eval_lengths):7.1f} ± {np.std(eval_lengths):5.1f}")
print(f"  Upright Rate: {upright_count}/20 ({upright_count*5}%)")

env_eval.close()

# Generate video
print(f"\n Generating video...")

env_video = gym.make('Ant-v5', render_mode='rgb_array')
_step_video = env_video.step
env_video.step = lambda a: (lambda o,r,t,tr,i: (o, symmetric_fixed_modifier(o,a,r,i), t, tr, i))(*_step_video(a))

obs, _ = env_video.reset()
frames = []
video_reward = 0

for step in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = env_video.step(action)
    video_reward += reward
    frames.append(env_video.render())
    
    if done or truncated:
        break

# Save video locally and to Drive
imageio.mimsave('symmetric_fixed_gait.mp4', frames, fps=30)
imageio.mimsave(f'{backup_folder}/symmetric_fixed_gait.mp4', frames, fps=30)

print(f"  ✅ Video saved: symmetric_fixed_gait.mp4")
print(f"     Duration: {len(frames)/30:.1f}s, Reward: {video_reward:.0f}")

env.close()
env_video.close()

# Training summary
training_time = (time.time() - start_time) / 60

print("\n" + "="*70)
print(" SYMMETRIC_FIXED COMPLETE!")
print("="*70)
print(f" Training time: {training_time:.1f} minutes")
print(f" Test reward: {np.mean(eval_rewards):.0f}")
print(f" Upright: {upright_count}/20 episodes ({upright_count*5}%)")
print(f" Saved to: {backup_folder}/symmetric_fixed_FINAL.zip")
print("="*70)

# Download video
from google.colab import files
files.download('symmetric_fixed_gait.mp4')

print("\n✅ Video downloaded! Watch to confirm it stays upright!")
