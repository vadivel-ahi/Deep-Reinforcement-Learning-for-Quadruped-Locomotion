# ===== RECONNECT AND FINISH REMAINING VARIANTS =====

print("Installing packages...")
!pip install gymnasium[mujoco] stable-baselines3[extra] imageio -q > /dev/null 2>&1

# Reconnect to Drive
from google.colab import drive
drive.mount('/content/drive')

import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import time
from datetime import datetime

# Use SAME backup folder
backup_folder = '/content/drive/MyDrive/RL_RewardShaping_20260411_1333'

print("RESUMING TRAINING")
print(f"Using existing folder: {backup_folder}")
print("\n✅ Already completed:")
print("   - baseline (2204 reward)")
print("   - symmetric (599 reward)")
print("\nTraining remaining:")
print("   - efficient")
print("   - speed")
print("="*70 + "\n")

def train_1M(name, reward_mod=None):
    print(f"\n{name.upper()}")
    
    env = gym.make('Ant-v5')
    
    if reward_mod:
        _step = env.step
        env.step = lambda a: (lambda o,r,t,tr,i: (o, reward_mod(o,a,r,i), t, tr, i))(*_step(a))
    
    model = PPO("MlpPolicy", env, verbose=0, device='cpu')
    
    start = time.time()
    print("Training 1,000,000 steps in 10 chunks...")
    
    for i in range(10):
        print(f"  Chunk {i+1}/10 ({(i+1)*10}%)...", end=' ', flush=True)
        model.learn(100_000, reset_num_timesteps=False, progress_bar=False)
        model.save(f"{backup_folder}/{name}_ckpt{i+1}")
        print(f"✅")
    
    model.save(f"{name}_FINAL")
    model.save(f"{backup_folder}/{name}_FINAL")
    
    print(f"\nSaved to Drive: {backup_folder}/{name}_FINAL.zip ✅")
    
    # Test
    obs, _ = env.reset()
    r = 0
    for _ in range(1000):
        a, _ = model.predict(obs, deterministic=True)
        obs, rew, d, t, _ = env.step(a)
        r += rew
        if d or t: break
    
    print(f"✅ {name} DONE! Test: {r:.0f} reward\n")
    
    env.close()
    return {'variant': name, 'test_reward': float(r)}

# Reward modifiers
eff = lambda o,a,r,i: r - 0.3*np.sum(a**2)
spd = lambda o,a,r,i: 1.3*i.get('x_velocity',0) - 0.4*np.sum(a**2) + 1.0

# TRAIN REMAINING VARIANTS
results = []
results.append(train_1M('efficient', eff))
results.append(train_1M('speed', spd))

print("\n" + "="*70)
print("✅ REMAINING VARIANTS COMPLETE!")
print("="*70)
for r in results:
    print(f"  {r['variant']}: {r['test_reward']:.0f}")

print(f"\nAll 4 models now in: {backup_folder}")
print("="*70)
