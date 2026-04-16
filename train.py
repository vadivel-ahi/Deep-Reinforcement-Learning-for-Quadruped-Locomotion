# ===== COMPLETE: INSTALL + TRAIN + SAVE TO DRIVE =====

# Step 1: Install packages
print("Installing packages...")
!pip install gymnasium[mujoco] stable-baselines3[extra] imageio -q > /dev/null 2>&1

# Step 2: Set environment
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# Step 3: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Step 4: Imports
import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import time
import json
from datetime import datetime

# Step 5: Create backup folder
backup_folder = f'/content/drive/MyDrive/RL_RewardShaping_{datetime.now().strftime("%Y%m%d_%H%M")}'
os.makedirs(backup_folder, exist_ok=True)

print("\n" + "="*70)
print("1M TRAINING WITH GOOGLE DRIVE PROTECTION")
print("="*70)
print(f" Backup: {backup_folder}")
print(f" Start: {datetime.now().strftime('%H:%M:%S')}")
print(f" Estimated: ~2 hours total")
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
        chunk_start = time.time()
        
        model.learn(100_000, reset_num_timesteps=False, progress_bar=False)
        
        # Save checkpoint to Drive every 100k
        model.save(f"{backup_folder}/{name}_ckpt{i+1}")
        
        print(f"✅ ({(time.time()-chunk_start)/60:.1f}min)")
    
    # Save final to BOTH locations
    model.save(f"{name}_FINAL")  # Local
    model.save(f"{backup_folder}/{name}_FINAL")  # Drive
    
    print(f"\nSaved:")
    print(f"   Local: {name}_FINAL.zip")
    print(f"   Drive: {backup_folder}/{name}_FINAL.zip ✅")
    
    # Quick test
    obs, _ = env.reset()
    r = 0
    for _ in range(1000):
        a, _ = model.predict(obs, deterministic=True)
        obs, rew, d, t, _ = env.step(a)
        r += rew
        if d or t: break
    
    elapsed = (time.time() - start) / 60
    
    print(f"\n✅ {name} DONE!")
    print(f"   Time: {elapsed:.1f}min ({elapsed/60:.2f}hr)")
    print(f"   Test: {r:.0f} reward\n")
    
    env.close()
    
    return {
        'variant': name,
        'time_min': elapsed,
        'test_reward': float(r),
        'done_at': datetime.now().strftime('%H:%M:%S')
    }

# Reward modifiers
sym = lambda o,a,r,i: r + 0.05*np.exp(-np.sum(np.abs(np.concatenate([o[7:9],o[11:13]])-np.concatenate([o[9:11],o[13:15]]))))
eff = lambda o,a,r,i: r - 0.3*np.sum(a**2)
spd = lambda o,a,r,i: 1.3*i.get('x_velocity',0) - 0.4*np.sum(a**2) + 1.0

# TRAIN ALL
results = []
results.append(train_1M('baseline', None))
results.append(train_1M('symmetric', sym))
results.append(train_1M('efficient', eff))
results.append(train_1M('speed', spd))

# Save summary to Drive
with open(f'{backup_folder}/summary.json', 'w') as f:
    json.dump(results, f, indent=2)

print("="*70)
print(" ALL COMPLETE!")
print("="*70)
print(f" End: {datetime.now().strftime('%H:%M:%S')}")
print(f"\n Results:")
for r in results:
    print(f"  {r['variant']}: {r['test_reward']:.0f} reward ({r['time_min']:.0f}min)")

print(f"\n✅ Everything saved to: {backup_folder}")
print("="*70)
