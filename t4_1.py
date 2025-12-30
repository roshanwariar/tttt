import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from matplotlib.colors import Normalize

# 1. SETUP & MODEL LOADING
# ---------------------------------------------------------
model_id = "Qwen/Qwen3-4B-Instruct-2507"

print(f"Loading {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# 2. DEFINE THE "ATLAS" DATASET
prompts_map = {
    "Math": [
        "Solve: 15 * 12 + 4 =",
        "Calculate the area of a circle with radius 5:",
        "If 3x + 5 = 20, then x is",
        "The square root of 144 is",
        "Integrate 2x dx from 0 to 5"
    ],
    "Code": [
        "def fibonacci(n):",
        "print('Hello World')",
        "import numpy as np",
        "for i in range(10):",
        "class Transformer(nn.Module):"
    ],
    "History/Facts": [
        "The capital of France is",
        "The first US president was",
        "World War II ended in",
        "Water boils at 100 degrees",
        "The largest planet is"
    ],
    "Creative/Poetry": [
        "The moon whispered to the",
        "Once upon a time in a land of",
        "Write a haiku about leaves:",
        "Her eyes were like deep",
        "In the silence of the night,"
    ],
    "Logic/Syllogism": [
        "All men are mortal. Socrates is a man. Therefore,",
        "If it rains, the grass gets wet. It rained. Thus,",
        "A > B and B > C, so A is",
        "Either Red or Blue. Not Red. Therefore,",
        "All birds have wings. A penguin is a bird. So,"
    ]
}

colors = {
    "Math": "red", 
    "Code": "black", 
    "History/Facts": "blue", 
    "Creative/Poetry": "purple", 
    "Logic/Syllogism": "orange"
}

# 3. RUN SIMULATION
trajectories = []
labels = []
physics_metrics = [] 

print("Mapping the Manifold (Excluding Final Layer)...")

with torch.no_grad():
    for category, prompts in prompts_map.items():
        for prompt in prompts:
            # Run Model
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            outputs = model(**inputs, output_hidden_states=True)
            
            # Extract Trajectory of LAST token
            # CHANGE IS HERE: We slice [:-1] to exclude the final layer
            last_token_idx = inputs.input_ids.shape[1] - 1
            
            # outputs.hidden_states is a tuple. [:-1] grabs everything EXCEPT the last layer.
            hidden_states_no_final = outputs.hidden_states[:-1]
            
            traj = torch.stack(hidden_states_no_final).squeeze(1)[:, last_token_idx, :].cpu().numpy()
            
            trajectories.append(traj)
            labels.append(category)
            
            # --- CALCULATE PHYSICS METRICS ---
            velocities = traj[1:] - traj[:-1]
            arc_length = np.sum(np.linalg.norm(velocities, axis=1))
            
            v_norm = velocities / (np.linalg.norm(velocities, axis=1, keepdims=True) + 1e-9)
            cosines = np.sum(v_norm[:-1] * v_norm[1:], axis=1)
            cosines = np.clip(cosines, -1.0, 1.0)
            angles = np.arccos(cosines) * (180 / np.pi) 
            total_curvature = np.sum(angles)
            
            physics_metrics.append((arc_length, total_curvature))

# 4. PLOT 1: THE MAP (PCA Trajectories)
combined_data = np.vstack(trajectories)
pca = PCA(n_components=2)
pca.fit(combined_data)

plt.figure(figsize=(18, 8))

# Subplot 1: The Trajectories
plt.subplot(1, 2, 1)
for i, traj in enumerate(trajectories):
    cat = labels[i]
    color = colors[cat]
    
    traj_2d = pca.transform(traj)
    
    lbl = cat if (labels.index(cat) == i) else None
    plt.plot(traj_2d[:, 0], traj_2d[:, 1], color=color, alpha=0.5, label=lbl)
    
    # Start point (Model Embedding)
    plt.scatter(traj_2d[0, 0], traj_2d[0, 1], color=color, s=20, marker='o') 
    # End point (Last Internal Layer)
    plt.scatter(traj_2d[-1, 0], traj_2d[-1, 1], color=color, s=40, marker='x') 

plt.title("The Atlas of Thought (Final Layer Excluded)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

