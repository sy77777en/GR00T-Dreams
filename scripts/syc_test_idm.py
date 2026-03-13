import torch
import numpy as np
from omegaconf import OmegaConf
from hydra.utils import instantiate
 
 
def sep(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
 
 
def show(label, tensor, note=""):
    if isinstance(tensor, torch.Tensor):
        info = f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}"
    elif isinstance(tensor, np.ndarray):
        info = f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}"
    else:
        info = str(tensor)
    n = f"  # {note}" if note else ""
    print(f"    {label:35s} {info}{n}")
    
 # ============================================================
sep("STEP 0: Load model from base.yaml")
# ============================================================
 
# Load the IDM model using Hydra (same as idm_training.py does)
cfg = OmegaConf.load("IDM_dump/base.yaml")
model = instantiate(cfg)
 
print(f"  Model type: {type(model)}")
print(f"  Model device: {next(model.parameters()).device}")
print(f"  action_dim: {model.action_dim}")
print(f"  action_horizon: {model.action_horizon}")   

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
print(f"  Moved to: {device}")


# ============================================================
sep("STEP 1: Construct fake input data (what transforms+collate produce)")
# ============================================================
 
B = 2           # batch size
n_frames = 2   # frames per sample
n_views = 1    # single view
# D_act_max = 32  # max action dim (padded)
D_act = 7       # actual action dim
T = 16          # action horizon
D_cam = 16      # camera pose dim
D_eef = 6       # eef state dim
MAX_SEQ = 112   # max VL sequence length
N_vtok = 16     # visual tokens per image after downsample

# Total images = B * n_frames * n_views = 2 * 2 * 1 = 4
N_IMG = B * n_frames * n_views

# 1. Images: SigLIP-processed, concatenated across batch
#    Real: siglip_processor.image_processor outputs (N, 3, 256, 256) float32
images = torch.randn(N_IMG, 3, 256, 256, device=device)
show("images", images, f"{B} samples × {n_frames} frames, concatenated")

# 2. Camera poses: per-image, concatenated like images
camera_poses = torch.randn(N_IMG, D_cam, device=device)
show("camera_poses", camera_poses, "per-image extrinsic 4x4 flattened")
 
# 3. EEF state: per-sample, stacked
eef_state = torch.randn(B, D_eef, device=device)
show("eef_state", eef_state, "initial robot state")

# 4. Actions: padded to max_action_dim (training only)
actions = torch.randn(B, T, D_act, device=device) 
show("actions", actions, f"GT actions padded {D_act}")

# 5. Actions mask
actions_mask = torch.ones(B, T, D_act, dtype=torch.bool, device=device)  
actions_mask[:, :, :D_act] = True
show("actions_mask", actions_mask, f"first {D_act} dims True")

# 6. VL token IDs: [PAD...PAD, IMG...IMG]
n_img_tokens_per_sample = n_frames * n_views * N_vtok  # 2 * 1 * 16 = 32
vl_token_ids = torch.zeros(B, MAX_SEQ, dtype=torch.long, device=device)
vl_token_ids[:, -n_img_tokens_per_sample:] = 1  # IMG_TOKEN = 1
show("vl_token_ids", vl_token_ids, f"[PAD×{MAX_SEQ - n_img_tokens_per_sample}, IMG×{n_img_tokens_per_sample}]")
 
# 7. SA token IDs: [ACT...ACT]
sa_token_ids = torch.full((B, T), 4, dtype=torch.long, device=device)  # ACT_TOKEN = 4
show("sa_token_ids", sa_token_ids, f"[ACT×{T}]")
 
# 8. VL attention mask
vl_attn_mask = torch.zeros(B, MAX_SEQ, dtype=torch.bool, device=device)
vl_attn_mask[:, -n_img_tokens_per_sample:] = True
show("vl_attn_mask", vl_attn_mask, f"last {n_img_tokens_per_sample} True")
 
# 9. Embodiment ID
embodiment_id = torch.full((B,), 17, dtype=torch.long, device=device)  # franka
show("embodiment_id", embodiment_id, "franka=17")

# Pack into the dict that the model expects
batch_dict = {
    "images": images,
    "camera_poses": camera_poses,
    "eef_state": eef_state,
    "actions": actions,
    "actions_mask": actions_mask,
    "vl_token_ids": vl_token_ids,
    "sa_token_ids": sa_token_ids,
    "vl_attn_mask": vl_attn_mask,
    "embodiment_id": embodiment_id,
}

# ============================================================
sep("STEP 2: Trace IDM.prepare_input()  [idm.py]")
# ============================================================
 
print("\n  Calling: model.prepare_input(batch_dict)")
print("    → validate_inputs()")
print("    → backbone.prepare_input()")
print("    → action_head.prepare_input()")
print("    → tree.map_structure (move to GPU + cast dtype)")

backbone_inputs, action_inputs = model.prepare_input(batch_dict)
print(f"\n  backbone_inputs keys: {list(backbone_inputs.keys())}")
for k, v in backbone_inputs.items():
    show(f"  backbone_inputs[{k}]", v)
 
print(f"\n  action_inputs keys: {list(action_inputs.keys())}")
for k, v in action_inputs.items():
    show(f"  action_inputs[{k}]", v)


# ============================================================
sep("STEP 3: Trace backbone.forward()  [identity.py]")
# ============================================================
 
print("\n  Calling: model.backbone(backbone_inputs)")
backbone_outputs = model.backbone(backbone_inputs)
 
for k, v in backbone_outputs.items():
    show(f"  backbone_outputs[{k}]", v, "empty tensor, just interface")


# ============================================================
sep("STEP 4: Trace action_head — encode_images()")
# ============================================================
 
action_head = model.action_head
data = action_inputs
 
print("\n  Calling: action_head.encode_images(data['images'], data['camera_poses'])")
print()

# Step by step inside encode_images:
print("  --- 4.1a: SigLIP vision encoder ---")
with torch.no_grad():
    siglip_out = action_head.siglip_model.vision_model(data["images"])["last_hidden_state"]
show("siglip output", siglip_out, "(256/16)^2 = 256 patch tokens")
 
print("\n  --- 4.1b: vision_projector ---")
img_feat = action_head.vision_projector(siglip_out)
show("after vision_proj", img_feat)
 
print("\n  --- 4.1c: mm_projector (double downsample) ---")
img_feat = action_head.mm_projector(img_feat)
show("after mm_projector", img_feat, "256→64→16 tokens")
 
print("\n  --- 4.1d: camera_pose_projector ---")
cam_emb = action_head.camera_pose_projector(data["camera_poses"])
show("cam_emb", cam_emb)
cam_emb_expanded = cam_emb.unsqueeze(1).expand(-1, img_feat.shape[1], -1)
show("cam_emb expanded", cam_emb_expanded)
visual_features = img_feat + cam_emb_expanded
show("visual_features (final)", visual_features)
 
# Also call the full function to verify
with torch.no_grad():
    visual_features_full = action_head.encode_images(data["images"], data["camera_poses"])
show("encode_images() output", visual_features_full, "should match above")

# ============================================================
sep("STEP 5: Trace action_head — noise injection + action_encoder")
# ============================================================
 
print("\n  --- 5.1: Flow matching noise ---")
actions_gt = data["actions"]
show("actions (GT)", actions_gt)
 
noise = torch.randn_like(actions_gt)
show("noise", noise)
 
t = action_head.sample_time(B, device=actions_gt.device, dtype=actions_gt.dtype)
show("t (sampled)", t)
t = t[:, None, None]
show("t (broadcast shape)", t)
 
noisy_trajectory = (1 - t) * noise + t * actions_gt
show("noisy_trajectory", noisy_trajectory)
 
velocity = actions_gt - noise
show("velocity (target)", velocity)
 
t_discretized = (t[:, 0, 0] * action_head.num_timestep_buckets).long()
show("t_discretized", t_discretized)
 
print("\n  --- 5.2: action_encoder ---")
with torch.no_grad():
    action_features = action_head.action_encoder(noisy_trajectory, t_discretized, data["embodiment_id"])
show("action_features", action_features)
 
print("\n  --- 5.3: eef_state injection ---")
with torch.no_grad():
    eef_feat = action_head.eef_state_projector(data["eef_state"])
show("eef_feat", eef_feat)
eef_feat_expanded = eef_feat.unsqueeze(1).expand(-1, action_features.shape[1], -1)
show("eef_feat expanded", eef_feat_expanded)
action_features = action_features + eef_feat_expanded
show("action_features + eef", action_features)

# ============================================================
sep("STEP 6: Trace action_head — prepare_input_embs()")
# ============================================================
 
print("\n  Calling: prepare_input_embs(vl_token_ids, sa_token_ids, visual_features, action_features)")
with torch.no_grad():
    vl_embs, sa_embs = action_head.prepare_input_embs(
        data["vl_token_ids"], data["sa_token_ids"],
        visual_features_full, action_features,
    )
show("vl_embs", vl_embs, "visual tokens scattered into IMG positions")
show("sa_embs", sa_embs, "action tokens + position embedding")
 
# Check what's zeros and what's filled in vl_embs
vl_nonzero = (vl_embs.abs().sum(-1) > 0).float().sum(-1)
print(f"    vl_embs non-zero positions per sample: {vl_nonzero.tolist()} (expected {n_img_tokens_per_sample})")

# ============================================================
sep("STEP 7: Trace action_head — vl_self_attention")
# ============================================================
 
print("\n  Calling: action_head.vl_self_attention_model(vl_embs)")
with torch.no_grad():
    vl_embs_attn = action_head.vl_self_attention_model(vl_embs)
show("vl_embs after self-attn", vl_embs_attn)

# ============================================================
sep("STEP 8: Trace action_head — DiT")
# ============================================================
 
print("\n  Calling: action_head.model(sa_embs, vl_embs, timestep=t_discretized)")
print(f"    hidden_states (query):     {tuple(sa_embs.shape)}")
print(f"    encoder_hidden (kv):       {tuple(vl_embs_attn.shape)}")
print(f"    timestep:                  {tuple(t_discretized.shape)}")
 
with torch.no_grad():
    dit_output = action_head.model(
        hidden_states=sa_embs,
        encoder_hidden_states=vl_embs_attn,
        encoder_attention_mask=data["vl_attn_mask"],
        timestep=t_discretized,
    )
show("DiT output", dit_output)

# ============================================================
sep("STEP 9: Trace action_head — action_decoder + loss")
# ============================================================
 
print("\n  Calling: action_head.action_decoder(dit_output, embodiment_id)")
with torch.no_grad():
    pred = action_head.action_decoder(dit_output, data["embodiment_id"])
show("pred (raw)", pred)
 
pred_actions = pred[:, -T:]
show("pred_actions", pred_actions)
 
print("\n  --- Loss computation ---")
show("pred_actions", pred_actions)
show("velocity", velocity)
print(f"    ⚠️  pred is {pred_actions.shape[-1]}-dim, velocity is {velocity.shape[-1]}-dim")
print(f"    FIX: truncate velocity and mask")
 
velocity_trunc = velocity[:, :, :model.action_dim]
mask_trunc = data["actions_mask"][:, :, :model.action_dim].float()
show("velocity (truncated)", velocity_trunc)
show("mask (truncated)", mask_trunc)
 
raw_loss = torch.nn.functional.mse_loss(pred_actions, velocity_trunc, reduction="none")
show("raw_loss", raw_loss)
loss = (raw_loss * mask_trunc).sum() / mask_trunc.sum()
show("loss", loss)

# ============================================================
sep("STEP 10: Full forward pass (training)")
# ============================================================
 
print("\n  Calling: model.forward(batch_dict)")
print("  (This calls the entire pipeline in one shot)")
 
# Need to call action_head.forward directly since we bypass validate_inputs
model.train()
with torch.no_grad():
    try:
        output = model.action_head(backbone_outputs, action_inputs)
        print(f"\n  Output keys: {list(output.keys())}")
        for k, v in output.items():
            show(f"  output[{k}]", v)
            print(v)
    except Exception as e:
        print(f"\n  ⚠️  Error (expected if shape mismatch not fixed): {e}")
        print(f"  This confirms the velocity truncation fix is needed!")

# ============================================================
sep("STEP 11: Full inference pass (get_action)")
# ============================================================
 
print("\n  Calling: model.action_head.get_action(backbone_outputs, action_inputs)")
model.eval()
with torch.no_grad():
    try:
        output = model.action_head.get_action(backbone_outputs, action_inputs)
        print(f"\n  Output keys: {list(output.keys())}")
        for k, v in output.items():
            show(f"  output[{k}]", v)
            print(v.shape)
    except Exception as e:
        print(f"\n  ⚠️  Error: {e}")
        import traceback
        traceback.print_exc()


# ============================================================
sep("STEP 12: Trainable parameters summary")
# ============================================================
 
total_params = 0
trainable_params = 0
print(f"\n  {'Module':50s} {'Total':>12s} {'Trainable':>12s} {'Frozen':>12s}")
print(f"  {'-'*86}")
 
for name, module in [
    ("action_head.siglip_model", action_head.siglip_model),
    ("action_head.vision_projector", action_head.vision_projector),
    ("action_head.mm_projector", action_head.mm_projector),
    ("action_head.camera_pose_projector", action_head.camera_pose_projector),
    ("action_head.eef_state_projector", action_head.eef_state_projector),
    ("action_head.vl_self_attention_model", action_head.vl_self_attention_model),
    ("action_head.action_encoder", action_head.action_encoder),
    ("action_head.model (DiT)", action_head.model),
    ("action_head.action_decoder", action_head.action_decoder),
    ("action_head.position_embedding", action_head.position_embedding),
]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    frozen = total - trainable
    total_params += total
    trainable_params += trainable
    print(f"  {name:50s} {total:12,d} {trainable:12,d} {frozen:12,d}")
 
print(f"  {'-'*86}")
print(f"  {'TOTAL':50s} {total_params:12,d} {trainable_params:12,d} {total_params - trainable_params:12,d}")
print(f"\n  Trainable: {trainable_params/total_params*100:.1f}%")
 
 
sep("DONE")
print("\n  All shapes tracked successfully!")
print("  You can now see the exact tensor shapes at every step of the IDM pipeline.")
