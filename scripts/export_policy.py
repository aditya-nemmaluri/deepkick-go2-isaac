import os
import torch
import torch.nn as nn

# 1. Load Checkpoint
checkpoint_path = os.path.expanduser("~/deepkick-go2-isaac/checkpoints/model_1499.pt")
checkpoint = torch.load(checkpoint_path, map_location="cpu")

print(f"[INFO] Detected checkpoint keys: {list(checkpoint.keys())}")

# 2. Reconstruct Policy MLP Architecture (33 -> 512 -> 256 -> 128 -> 12)
actor = nn.Sequential(
    nn.Linear(33, 512),
    nn.ELU(),
    nn.Linear(512, 256),
    nn.ELU(),
    nn.Linear(256, 128),
    nn.ELU(),
    nn.Linear(128, 12)
)

# 3. Handle key structure across RSL-RL versions
if "actor_state_dict" in checkpoint:
    raw_dict = checkpoint["actor_state_dict"]
elif "model_state_dict" in checkpoint:
    raw_dict = checkpoint["model_state_dict"]
else:
    raw_dict = checkpoint

# Clean layer name prefixes ('actor.mlp.0.weight' -> '0.weight')
clean_dict = {}
for k, v in raw_dict.items():
    k_clean = k.replace("actor.", "").replace("mlp.", "")
    if k_clean in actor.state_dict():
        clean_dict[k_clean] = v

actor.load_state_dict(clean_dict)
actor.eval()

# 4. Export Standalone PyTorch & ONNX Artifacts
out_pt = os.path.expanduser("~/deepkick-go2-isaac/checkpoints/go2_kick_actor.pt")
out_onnx = os.path.expanduser("~/deepkick-go2-isaac/checkpoints/go2_kick_policy.onnx")

torch.save(actor, out_pt)

dummy_input = torch.randn(1, 33)
torch.onnx.export(
    actor,
    dummy_input,
    out_onnx,
    input_names=["observation"],
    output_names=["action_joint_targets"],
    dynamic_axes={"observation": {0: "batch_size"}, "action_joint_targets": {0: "batch_size"}}
)

print(f"[SUCCESS] Exported standalone policy weights to:\n - {out_pt}\n - {out_onnx}")
