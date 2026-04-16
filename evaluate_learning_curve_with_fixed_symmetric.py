# ===== LEARNING CURVES - ALL 5 VARIANTS (CORRECTED) =====

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

backup_folder = '/content/drive/MyDrive/RL_RewardShaping_20260411_1333'

# Load data
df = pd.read_csv(f'{backup_folder}/learning_curves_complete.csv')

# Fix: Combine 'Mean_Reward' and 'Reward' columns
# Use Mean_Reward if available, otherwise use Reward
df['Reward_Final'] = df['Mean_Reward'].fillna(df['Reward'])

print("="*70)
print("📈 LEARNING CURVES - ALL 5 VARIANTS")
print("="*70)
print(f"\nVariants: {sorted(df['Variant'].unique())}")
print("="*70 + "\n")

# Create figure
plt.figure(figsize=(14, 8))

# Colors and markers
colors = {
    'baseline': '#3498db',
    'symmetric': '#e74c3c',
    'efficient': '#2ecc71',
    'speed': '#f39c12',
    'symmetric_fixed': '#9b59b6'
}

markers = {
    'baseline': 'o',
    'symmetric': 's',
    'efficient': '^',
    'speed': 'D',
    'symmetric_fixed': 'p'
}

# Plot each variant
for variant_name in ['Baseline', 'Symmetric', 'Efficient', 'Speed', 'symmetric_fixed']:
    data = df[df['Variant'] == variant_name].sort_values('Timesteps')
    
    if len(data) > 0:
        timesteps = data['Timesteps'].values
        rewards = data['Reward_Final'].values  # Use combined column
        
        # Display name
        if variant_name == 'symmetric_fixed':
            label_name = 'Symmetric_Fixed'
            color_key = 'symmetric_fixed'
            marker_key = 'symmetric_fixed'
        else:
            label_name = variant_name
            color_key = variant_name.lower()
            marker_key = variant_name.lower()
        
        # Plot
        plt.plot(timesteps, rewards,
                marker=markers[marker_key],
                linewidth=3,
                markersize=10,
                color=colors[color_key],
                label=label_name,
                alpha=0.9)
        
        print(f"✅ {label_name}: {len(data)} points, final={rewards[-1]:.0f}")

# Formatting
plt.xlabel('Training Timesteps', fontsize=15, fontweight='bold')
plt.ylabel('Mean Episode Reward', fontsize=15, fontweight='bold')
plt.title('Learning Curves: Training Progression (All Variants)',
          fontsize=17, fontweight='bold', pad=20)

plt.legend(fontsize=13, loc='lower right', framealpha=0.95,
          edgecolor='black', fancybox=True)

plt.grid(True, alpha=0.3, linestyle='--', linewidth=1)
plt.axhline(y=0, color='black', linewidth=1.2, alpha=0.6)

# Format axes
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)

plt.xticks([0, 200_000, 400_000, 600_000, 800_000, 1_000_000],
          ['0', '200K', '400K', '600K', '800K', '1M'],
          fontsize=12)
plt.yticks(fontsize=12)

plt.xlim(-30000, 1_030_000)

ax.set_facecolor('#fafafa')

plt.tight_layout()

# Save
plt.savefig('learning_curves_all5.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(f'{backup_folder}/learning_curves_all5.png', dpi=300, bbox_inches='tight', facecolor='white')

plt.show()

print("\n✅ Learning curves plot created!")

# Download
from google.colab import files
files.download('learning_curves_all5.png')

print("\n🎉 DONE! All 5 variants plotted!")
