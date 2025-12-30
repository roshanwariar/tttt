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

prompts_map = {
    # --- MAIN ---
    "Pure Code": [
        "def quicksort(arr):",               
        "import torch.nn as nn",             
        "while True: print('loop')",         
    ],
    "Pure Math": [
        "15 * 44 + 12 =",                    
        "Calculate the integral of 2x:",     
        "Solve for x: 3x - 5 = 10",          
    ],
    "Pure Story": [
        "Once upon a time in a castle",      
        "The sun set over the horizon",      
        "She whispered into the wind",       
    ],

    # --- INBETWEEN ---
    "Word Problems": [ 
        "If John has 50 apples and gives 12 to Mary, he has", 
        "A train leaves Chicago at 60mph. Two hours later,",  
        "Mary is twice as old as John. If John is 10, Mary is" 
    ],
    "Docstrings": [ 
        "This function calculates the fibonacci sequence by", # English-heavy
        "# The following code connects to the database and",  # Comment style
        "''' This class handles user authentication '''"      # Code-syntax style
    ],
    "Pseudocode": [ 
        "IF user is valid THEN grant access ELSE",            
        "REPEAT until x is greater than 10:",                 
        "SET counter to zero and INCREMENT by one"            
    ],
    "Algorithmic Poetry": [ 
        "class Love(Emotion): def __init__(self):",           
        "while heart.is_beating(): print('I miss you')",      
        "if (tears > 0): return sadness else: return hope"    
    ]
}

colors = {
    "Pure Code": "black", 
    "Pure Math": "red", 
    "Pure Story": "blue",
    "Word Problems": "magenta",
    "Docstrings": "teal",
    "Pseudocode": "grey",
    "Algorithmic Poetry": "purple"
}

# 3. RUN SIMULATION
trajectories = []
labels = []        
line_numbers = []  # To track 1, 2, 3... for the plot
global_counter = 1 # Start counting from 1

print("Mapping with Numbered Lines...")

with torch.no_grad():
    for category, prompts in prompts_map.items():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            outputs = model(**inputs, output_hidden_states=True)
            
            # Slice [:-1] to exclude final LayerNorm (Crucial for clear plot)
            last_token_idx = inputs.input_ids.shape[1] - 1
            hidden_states_no_final = outputs.hidden_states[:-1]
            
            traj = torch.stack(hidden_states_no_final).squeeze(1)[:, last_token_idx, :].cpu().numpy()
            
            trajectories.append(traj)
            labels.append(category)
            line_numbers.append(global_counter)
            global_counter += 1

# 4. PLOT
combined_data = np.vstack(trajectories)
pca = PCA(n_components=2)
pca.fit(combined_data)

plt.figure(figsize=(14, 12))

# Keep track of handles for a clean legend (one entry per category)
handles = {} 

for i, traj in enumerate(trajectories):
    cat = labels[i]
    num = line_numbers[i]
    color = colors[cat]
    
    is_anchor = "Pure" in cat
    style = '-' if is_anchor else '--'
    width = 1.5 if is_anchor else 2.5
    alpha = 0.4 if is_anchor else 0.9 
    
    traj_2d = pca.transform(traj)
    
    # Plot the line
    line, = plt.plot(traj_2d[:, 0], traj_2d[:, 1], color=color, linestyle=style, linewidth=width, alpha=alpha)
    
    # Add NUMBER text near the END of the line
    # We use the end point (traj_2d[-1]) so you see where the thought "landed"
    plt.text(traj_2d[-1, 0] + 2, traj_2d[-1, 1] + 2, str(num), fontsize=12, fontweight='bold', color=color)
    
    # Mark the end
    marker = 'o' if is_anchor else '*'
    plt.scatter(traj_2d[-1, 0], traj_2d[-1, 1], color=color, marker=marker, s=80)
    
    if cat not in handles:
        handles[cat] = line

plt.legend(handles.values(), handles.keys(), loc='best')
plt.title("The Spectrum of Thought (Numbered)")
plt.xlabel("PC1 (Time/Depth)")
plt.ylabel("PC2 (Modality/Syntax)")
plt.grid(True, alpha=0.3)

# --- PRINT THE KEY ---
print("\n" + "="*40)
print("      LEGEND KEY: PROMPT ID MAP      ")
print("="*40)
counter = 1
for category, prompts in prompts_map.items():
    print(f"\n[{category}]")
    for prompt in prompts:
        # Highlight the Docstrings for easier checking
        prefix = ">>> " if category == "Docstrings" else "    "
        print(f"{prefix}{counter}: {prompt}")
        counter += 1
print("="*40)

plt.show()
