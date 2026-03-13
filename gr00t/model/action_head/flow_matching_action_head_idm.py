from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature


_PAD_TOKEN = 0
_IMG_TOKEN = 1
_IMG_SEP_TOKEN = 5  # New separator token
_LANG_TOKEN = 2
_PROPRIO_TOKEN = 3
_ACT_TOKEN = 4

def swish(x):
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Produces a sinusoidal encoding of shape (B, T, w)
    given timesteps of shape (B, T).
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        # timesteps: shape (B, T)
        # We'll compute sin/cos frequencies across dim T
        timesteps = timesteps.float()  # ensure float

        B, T = timesteps.shape
        device = timesteps.device

        half_dim = self.embedding_dim // 2
        # typical log space frequencies for sinusoidal encoding
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        # Expand timesteps to (B, T, 1) then multiply
        freqs = timesteps.unsqueeze(-1) * exponent.exp()  # (B, T, half_dim)

        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        enc = torch.cat([sin, cos], dim=-1)  # (B, T, w)

        return enc


@dataclass
class FlowMatchingActionHeadIDMConfig(PretrainedConfig):
    """
        Siyuan Cen on 2026-03-12
        Add camera pose embedding.
    """
    camera_pose_dim: int = field(default=16, metadata={"help": "Flattened camera pose dim"})
    eef_state_dim: int = field(default=16, metadata={"help": "eef initial state dim (padded)"})
    add_eef_state_embed: bool = field(default=True, metadata={"help": "Whether to add eef initial state embedding"})
    add_camera_pose_embed: bool = field(default=True, metadata={"help": "Whether to add camera pose embedding"})
    
    add_pos_embed: bool = field(
        default=False, metadata={"help": "Whether to add positional embedding"}
    )
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."})
    model_type: str = field(
        default="diffusion_policy_head", metadata={"help": "Model type identifier."}
    )
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    mm_projector_cfg: dict = field(
        default=None, metadata={"help": "Multimodal Projector configuration."}
    )
    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    siglip_hidden_size: int = field(default=768, metadata={"help": "Siglip hidden size."})
    add_view_embed: bool = field(default=False, metadata={"help": "Whether to add view embedding."})
    max_num_views: int = field(default=3, metadata={"help": "Maximum number of views."})

    tune_vision_tower: bool = field(default=True, metadata={"help": "Tune vision if True."})
    tune_mm_projector: bool = field(default=True, metadata={"help": "Tune mm projector if True."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Tune diffusion model if True."}
    )
    tune_multi_projector: bool = field(
        default=True, metadata={"help": "Tune multi projector if True."}
    )
    tune_vl_mixing: bool = field(default=True, metadata={"help": "Tune vl mixing if True."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


class FlowMatchingActionHeadIDM(nn.Module):
    config_class = FlowMatchingActionHeadIDMConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: FlowMatchingActionHeadIDMConfig,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.siglip_model = instantiate(config.siglip_model_cfg)

        del self.siglip_model.text_model
        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.vision_projector = nn.Linear(self.config.siglip_hidden_size, self.hidden_size)
        self.model = instantiate(config.diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps
        self.vl_self_attention_model = instantiate(config.vl_self_attention_cfg)
        self.eef_state_projector = nn.Sequential(
            nn.Linear(config.eef_state_dim, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.hidden_size,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        self.mm_vision_select_layer = config.mm_vision_select_layer
        if config.mm_projector_cfg is not None:
            self.mm_projector = instantiate(config.mm_projector_cfg)
        else:
            self.mm_projector = None
        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.hidden_size)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        # if config.add_view_embed:
        #     self.view_embedding = nn.Embedding(config.max_num_views, self.hidden_size)
        #     nn.init.normal_(self.view_embedding.weight, mean=0.0, std=0.02)
        if config.add_camera_pose_embed:
            self.camera_pose_projector = nn.Linear(config.camera_pose_dim, self.hidden_size)
            nn.init.normal_(self.camera_pose_projector.weight, mean=0.0, std=0.02)

        self.set_trainable_parameters(
            tune_multi_projector=config.tune_multi_projector,
            tune_diffusion_model=config.tune_diffusion_model,
            tune_vision_tower=config.tune_vision_tower,
            tune_mm_projector=config.tune_mm_projector,
            tune_vl_mixing=config.tune_vl_mixing,
        )

        print(
            "total number of parameters: %e",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def set_trainable_parameters(
        self,
        tune_multi_projector: bool = True,
        tune_diffusion_model: bool = True,
        tune_vision_tower: bool = True,
        tune_mm_projector: bool = True,
        tune_vl_mixing: bool = True,
    ):
        self.tune_multi_projector = tune_multi_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vision_tower = tune_vision_tower
        self.tune_mm_projector = tune_mm_projector
        self.tune_vl_mixing = tune_vl_mixing

        for param in self.parameters():
            param.requires_grad = True
        # Freeze unused parameters in siglip vision encoder
        self.siglip_model.logit_scale.requires_grad = False
        self.siglip_model.logit_bias.requires_grad = False
        for param in self.siglip_model.vision_model.encoder.layers[11].parameters():
            param.requires_grad = False
        for param in self.siglip_model.vision_model.head.parameters():
            param.requires_grad = False

        # Freeze parameters
        if not tune_multi_projector:
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
            # if self.config.add_view_embed:
            #     self.view_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        if not tune_vision_tower:
            self.siglip_model.vision_model.requires_grad_(False)
            self.vision_projector.requires_grad_(False)
        if self.mm_projector is not None and not tune_mm_projector:
            self.mm_projector.requires_grad_(False)
        if not tune_vl_mixing:
            self.vl_self_attention_model.requires_grad_(False)

        print(f"Tune action head multi_projector: {self.tune_multi_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        print(f"Tune action head vision tower: {self.tune_vision_tower}")
        print(f"Tune action head mm_projector: {self.tune_mm_projector}")
        print(f"Tune action head vl_mixing: {self.tune_vl_mixing}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_multi_projector:
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
                # if self.config.add_view_embed:
                #     self.view_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()
            if not self.tune_vision_tower:
                self.siglip_model.vision_model.eval()
                self.vision_projector.eval()
            if self.mm_projector is not None and not self.tune_mm_projector:
                self.mm_projector.eval()
            if not self.tune_vl_mixing:
                self.vl_self_attention_model.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    """
        Siyuan Cen on 2026-03-12
        Encode images and camera poses.
    """
    def encode_images(self, images, camera_poses):
        image_features = self.siglip_model.vision_model(images)["last_hidden_state"]
        image_features = self.vision_projector(image_features)
        if self.mm_projector is not None:
            image_features = self.mm_projector(image_features)
        if self.config.add_camera_pose_embed:
            camera_pose_features = self.camera_pose_projector(camera_poses)
            camera_pose_features = camera_pose_features.unsqueeze(1).expand(-1, image_features.shape[1], -1)
            image_features = image_features + camera_pose_features
        # if self.config.add_view_embed:
        #     view_embs = self.view_embedding(view_ids)
            # view_embs = view_embs.unsqueeze(1).expand(-1, image_features.shape[1], -1)
            # image_features = image_features + view_embs
        return image_features

    def prepare_input_embs(self, vl_token_ids, sa_token_ids, vision, action):
        B, T = vl_token_ids.shape
        vl_embs = torch.full(
            size=(B, T, self.hidden_size), fill_value=0.0, dtype=vision.dtype, device=vision.device
        )

        # Project vision.
        vision_mask = vl_token_ids == _IMG_TOKEN
        vision_mask = vision_mask.unsqueeze(-1).expand_as(vl_embs)
        vl_embs = vl_embs.masked_scatter(vision_mask, vision)

        # Project image separator using the learnable sep_embedding.
        sep_mask = vl_token_ids == _IMG_SEP_TOKEN  # shape: (B, T)
        num_sep = sep_mask.sum().item()
        if num_sep > 0:
            # Expand the separator embedding for each occurrence.
            repeated_sep = self.vis_sep_embedding.unsqueeze(0).expand(num_sep, self.hidden_size)
            # Assign the separator embeddings to the correct positions.
            vl_embs[sep_mask] = repeated_sep.to(dtype=vl_embs.dtype)

        B, T = sa_token_ids.shape
        sa_embs = torch.full(
            size=(B, T, self.hidden_size), fill_value=0.0, dtype=vision.dtype, device=vision.device
        )

        # Project action.
        action_mask = sa_token_ids == _ACT_TOKEN
        action_mask = action_mask.unsqueeze(-1).expand_as(sa_embs)
        sa_embs = sa_embs.masked_scatter(action_mask, action)

        # Add positional embeddings
        pos_ids = torch.arange(T, dtype=torch.long, device=sa_token_ids.device)
        if self.config.add_pos_embed:
            pos_embs = self.position_embedding(pos_ids)  # (T, hidden_size)
            pos_embs = pos_embs.unsqueeze(0).expand(B, T, self.hidden_size)
            sa_embs = sa_embs + pos_embs
        return vl_embs, sa_embs

    # ========= ActionHead required ============
    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()

        data = action_input
        embodiment_id = action_input.embodiment_id
        # 1) Encode images/state
        # visual_features = self.encode_images(data["images"], data["view_ids"])
        visual_features = self.encode_images(data["images"], data["camera_poses"])

        # 2) Prepare noisy trajectory
        actions = data["actions"]
        noise = torch.randn_like(actions)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # 3) Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()

        # 4) Get action encoder embeddings with correct time argument
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)
        
        """
            Siyuan Cen on 2026-03-12
            Encode eef initial state.
        """
        eef_state = data["eef_state"]
        if self.config.add_eef_state_embed:
            eef_feat = self.eef_state_projector(eef_state)      # (B, hidden)
            eef_feat = eef_feat.unsqueeze(1).expand(-1, action_features.shape[1], -1)
            action_features = action_features + eef_feat
        
        # 5) Prepare full input to DiT (or your model)
        vl_embs, sa_embs = self.prepare_input_embs(
            data["vl_token_ids"],
            data["sa_token_ids"],
            visual_features,
            action_features,
        )
        vl_embs = self.vl_self_attention_model(vl_embs)
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=data["vl_attn_mask"],
            timestep=t_discretized,
        )
        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # 6) Flow-matching or velocity-prediction MSE
        #    Mask for variable-length trajectories
        mask = data["actions_mask"]  # shape => (B, seq_len_of_actions, ...)
        raw_loss = F.mse_loss(pred_actions, velocity, reduction="none")
        raw_loss = raw_loss * mask
        loss = raw_loss.sum() / mask.sum()

        return BatchFeature(data={"loss": loss})

    @torch.inference_mode()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        For i in [0..N-1]:
          1) t = i/N
          2) velocity = model(x(t), t)
          3) x(t + dt) = x(t) + dt * velocity
        """

        data = action_input
        embodiment_id = action_input.embodiment_id

        batch_size = embodiment_id.shape[0]
        device = data.images.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=data.images.dtype,
            device=device,
        )

        # 1) Hyperparameters for flow sampling
        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # 2) Encode static context (images, text, state) once if it does not depend on actions
        visual_features = self.encode_images(data["images"], data["camera_poses"])

        # 3) Start denoising the actions
        for i in range(num_steps):
            # ---- (a) Discretize continuous time in [0,1]
            t_cont = i / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # ---- (b) Build embeddings (actions included)
            # Pass the *current* actions at time t into the action encoder
            action_features = self.action_encoder(
                actions,
                (torch.ones(actions.shape[0]) * t_discretized).cuda().to(device),
                embodiment_id,
            )
            if self.config.add_eef_state_embed:
                eef_feat = self.eef_state_projector(data["eef_state"])
                eef_feat = eef_feat.unsqueeze(1).expand(-1, action_features.shape[1], -1)
                action_features = action_features + eef_feat
            vl_embs, sa_embs = self.prepare_input_embs(
                data["vl_token_ids"],
                data["sa_token_ids"],
                visual_features,
                action_features,
            )
            vl_embs = self.vl_self_attention_model(vl_embs)
            # ---- (c) Forward pass to get velocity = d/dt x(t)
            timesteps = torch.from_numpy(np.array([t_discretized])).to(device).long()
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                encoder_attention_mask=data["vl_attn_mask"],
                timestep=timesteps,
            )
            pred = self.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -actions.shape[1] :]

            # ---- (d) Naive Euler step: x(t + dt) = x(t) + dt * velocity
            actions = actions + dt * pred_velocity

        # 5) Return final actions at t=1
        return BatchFeature(data={"action_pred": actions})

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
