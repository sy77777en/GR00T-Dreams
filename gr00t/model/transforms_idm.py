# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import tree
from pydantic import Field, PrivateAttr
from transformers.data.data_collator import DataCollatorMixin

from gr00t.data.schema import DatasetMetadata, EmbodimentTag
from gr00t.data.transform.base import InvertibleModalityTransform
from gr00t.model.action_head.siglip import SiglipProcessor

from einops import rearrange
import PIL

# Set IDs for each token type.
_PAD_TOKEN = 0
_IMG_TOKEN = 1
_IMG_SEP_TOKEN = 5  # New separator token
_LANG_TOKEN = 2
_PROPRIO_TOKEN = 3
_ACT_TOKEN = 4


def collate(features) -> dict:
    batch = {}
    keys = features[0].keys()
    for key in keys:
        values = [elem[key] for elem in features]
        """
            Siyuan Cen on 2026-03-12
            Add camera poses (extrinsics) to the input data.
        """
        if key in ["images", "camera_poses"]:
            batch[key] = torch.from_numpy(np.concatenate(values))
        else:
            batch[key] = torch.from_numpy(np.stack(values))
    return batch


class DefaultDataCollatorGR00TIDM(DataCollatorMixin):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        return collate(features)


class GR00TIDMTransform(InvertibleModalityTransform):
    _EMBODIMENT_TAG_MAPPING = {
        "gr1": 24,
        "franka": 17,
        "so100": 26,
        "robocasa_panda_omron": 13,
        "new_embodiment": 31,  # use the last projector for new embodiment,
    }

    # -- We inherit from ModalityTransform, so we keep apply_to as well --
    apply_to: list[str] = Field(
        default_factory=list, description="Not used in this transform, kept for compatibility."
    )
    training: bool = Field(
        default=True, description="Whether to apply the transform in training mode."
    )
    embodiment_tag_mapping: dict[str, int] = Field(
        description="The projector index of each embodiment tag.",
        default=_EMBODIMENT_TAG_MAPPING,
    )
    language_dropout_prob: float = Field(
        default=0.0,
        description="Dropout probability for language.",
    )

    # XEmbDiT arguments
    default_instruction: str = Field(default="Perform the default behavior.")
    siglip_processor: SiglipProcessor = Field(default=SiglipProcessor.from_pretrained("google/siglip2-large-patch16-256"))
    num_visual_tokens_per_frame: int = Field(default=16)
    max_num_images_per_sequence: int = Field(default=6)
    max_action_dim: int
    max_sequence_length: int = Field(default=112)
    action_horizon: int = None
    embodiment_tag: EmbodimentTag | None = None
    
    """
        Siyuan Cen on 2026-03-12
        Add camera poses (extrinsics) and initial robot state (eef state) to the input data.
    """
    camera_pose_dim: int = Field(default=16)   # Flatten (4x4) camera pose matrix
    eef_state_dim: int = Field(default=6)      # Position (3) + Rotation (3)

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        """Set the metadata for the transform."""
        super().set_metadata(dataset_metadata)
        self.embodiment_tag = dataset_metadata.embodiment_tag

    def get_embodiment_tag(self) -> int:
        """Get the embodiment tag from the data."""
        assert (
            self.embodiment_tag is not None
        ), "Embodiment tag not set. Please call set_metadata first."
        return self.embodiment_tag_mapping[self.embodiment_tag.value]

    def check_keys_and_batch_size(self, data):
        grouped_keys = {}
        for key in data.keys():
            try:
                modality, _ = key.split(".")
            except:  # noqa: E722
                ### Handle language annotation special case
                if "annotation" in key:
                    modality = "language"
                else:
                    modality = "others"  # will contain the video, state, and action
            if modality not in grouped_keys:
                grouped_keys[modality] = []
            grouped_keys[modality].append(key)
        # Use video key to determine batch size.
        video_ndim = data["video"].ndim
        if video_ndim == 5:  # Interpret as [T, V, H, W, C]
            is_batched = False
            batch_size = 1
        elif video_ndim == 6:  # Interpret as [B, T, V, H, W, C]
            is_batched = True
            batch_size = data["video"].shape[0]
        else:
            raise ValueError(f"Unsupported video number of dimensions: {video_ndim}")

        return is_batched, batch_size

    def _prepare_video(self, data: dict):
        """Process, stack, and pad images from data['video']."""
        """
            Siyuan Cen on 2026-03-12
            Only consider one frame viewpoint for now. Shape should be [T, 1, H, W, C]
        """
        # view_ids = []
        images = rearrange(
            data["video"],
            "t v h w c -> (t v) h w c",
        )
        # for i in range(images.shape[0]):
        #     view_ids.append(i)
        # Pad to max_num_images_per_sequence
        n_images = min(self.max_num_images_per_sequence, images.shape[0])
        images = images[: self.max_num_images_per_sequence]
        n_image_tokens = n_images * self.num_visual_tokens_per_frame

        # Apply transform.
        processed_images = []
        for image_idx in range(images.shape[0]):
            img_data = PIL.Image.fromarray(images[image_idx])
            processed_images.append(
                self.siglip_processor.image_processor(images=[img_data])["pixel_values"]
            )
        images = np.concatenate(processed_images, axis=0)    # → (2, 3, 256, 256)
        return images, n_images, n_image_tokens


    def _prepare_camera_pose(self, data: dict, n_images: int):
        """
            Siyuan Cen on 2026-03-12
            Process camera poses (extrinsics) from data['camera_pose'].
            Shape should be (4, 4) Extrinsic matrix for two frames.
        """
        extrinsics = data["camera_extrinsics"]
        camera_poses = extrinsics.reshape(1, -1)  # (1, 16)
        
        if "camera_intrinsics" in data:
            intrinsics = data["camera_intrinsics"]  # (3, 3)
            intrinsics_flat = intrinsics.reshape(1, -1)  # (1, 9)
            camera_poses = np.concatenate([camera_poses, intrinsics_flat], axis=-1)  # (1, 25)
        camera_poses = np.repeat(camera_poses, repeats=n_images, axis=0)  # (n_images, D)
        return camera_poses
    
    
    def _prepare_eef_state(self, data: dict):
        """
            Siyuan Cen on 2026-03-12
            Process initial robot state (eef state) from data['state'].
            Shape should be (6,) Position (3) + Rotation (3)
        """
        eef_state = data["state.eef_initial_state"] # (6,)
        eef_init = eef_init.reshape(-1)
        state_dim = eef_init.shape[-1]
        eef_state = np.pad(eef_init, (0, self.max_state_dim - state_dim), "constant")   # (max_state_dim, )
        return eef_state
    
        
    def _prepare_action(self, data: dict):
        """
        Pad to max_action_dim, return masks.
        """
        actions = data["action"]

        n_action_tokens = actions.shape[0]  # T
        n_action_dims = actions.shape[1]

        assert (
            n_action_dims <= self.max_action_dim
        ), f"Action dim {n_action_dims} exceeds max allowed {self.max_action_dim}."

        # Pad the channel dimension
        actions = np.pad(actions, ((0, 0), (0, self.max_action_dim - n_action_dims)), "constant")

        # Create mask: [T, max_action_dim]
        actions_mask = np.zeros((n_action_tokens, self.max_action_dim), dtype=bool)
        actions_mask[:, :n_action_dims] = True

        return actions, actions_mask, n_action_tokens

    def _build_token_ids(self, n_images, n_action_tokens, has_state=False):
        """
        Build the 1D array of token_ids based on the number of each block.
        Return (token_ids, special_pad_token_idx).
        """
        vl_token_ids = []
        sa_token_ids = []

        # 1) Video placeholders
        for _ in range(n_images):
            vl_token_ids.extend([_IMG_TOKEN] * self.num_visual_tokens_per_frame)
            
        # State token
        if has_state:
            sa_token_ids.extend([_PROPRIO_TOKEN] * 1)

        # 5) Action tokens
        sa_token_ids.extend([_ACT_TOKEN] * n_action_tokens)

        return np.array(vl_token_ids), np.array(sa_token_ids)

    def _prepare_attention_mask(
        self,
        vl_token_ids: np.ndarray,
    ):
        """
        Build 1D attention mask for vision-language tokens.
        1 indicates valid token, 0 indicates padding token.
        State-action attention will be handled separately by the model.
        """
        # Only create attention mask for vision-language tokens
        vl_seq_len = vl_token_ids.shape[0]
        vl_attn_mask = np.ones(vl_seq_len, dtype=bool)  # All tokens are valid initially

        # Pad vl_token_ids and vl_attn_mask to max_sequence_length
        if vl_seq_len > self.max_sequence_length:
            raise ValueError("VL sequence length exceeds the max sequence length!")

        left_pad_len = self.max_sequence_length - vl_seq_len

        # Pad token_ids (with PAD_TOKEN)
        vl_token_ids = np.pad(vl_token_ids, (left_pad_len, 0), constant_values=_PAD_TOKEN)

        # Pad attention mask with 0 (padding tokens)
        vl_attn_mask = np.pad(vl_attn_mask, (left_pad_len, 0), constant_values=0)

        return vl_token_ids, vl_attn_mask

    ###########################################################################
    #                           apply / unapply
    ###########################################################################
    def apply_single(self, data: dict) -> dict:
        """
        Main entry point for the transform. We assume that `data` has
        data['video'], data['language'], data['state'], and data['action'] in
        the shapes needed. If you have multiple keys for each modality, you
        could use your own grouping logic (similar to GR1Transform) first.
        """
        self.check_keys_and_batch_size(data)
        transformed_data = {}

        # 1) Prepare video
        images, n_images, n_image_tokens = self._prepare_video(data)
        transformed_data["images"] = images
        # transformed_data["view_ids"] = view_ids
        
        """
            Siyuan Cen on 2026-03-12
            Add camera poses (extrinsics) and initial robot state (eef state) to the input data.
        """
        # 2) Prepare camera pose
        camera_poses = self._prepare_camera_pose(data, n_images)
        transformed_data["camera_poses"] = camera_poses
        
        # 3) Prepare eef state
        eef_state = self._prepare_eef_state(data)
        transformed_data["eef_state"] = eef_state


        if self.training:
            # 4) Prepare actions
            actions, actions_mask, n_action_tokens = self._prepare_action(data)
            transformed_data["actions"] = actions
            transformed_data["actions_mask"] = actions_mask
        else:
            n_action_tokens = self.action_horizon

        # 5) Build token_ids
        vl_token_ids, sa_token_ids = self._build_token_ids(n_images, n_action_tokens)

        # 6) Build the attention mask only for vision-language tokens
        vl_token_ids, vl_attn_mask = self._prepare_attention_mask(vl_token_ids)

        transformed_data["vl_token_ids"] = vl_token_ids
        transformed_data["sa_token_ids"] = sa_token_ids
        transformed_data["vl_attn_mask"] = vl_attn_mask
        transformed_data["embodiment_id"] = self.get_embodiment_tag()

        return transformed_data

    def apply_batch(self, data: dict, batch_size: int) -> dict:
        # Split on batch dimension.
        data_split = [tree.map_structure(lambda x: x[i], data) for i in range(batch_size)]
        # Process each element.
        data_split_processed = [self.apply_single(elem) for elem in data_split]
        return collate(data_split_processed)

    def apply(self, data: dict) -> dict:
        is_batched, batch_size = self.check_keys_and_batch_size(data)
        if is_batched:
            return self.apply_batch(data, batch_size)
        else:
            return self.apply_single(data)

    def unapply(self, data: dict) -> dict:
        # Leave as is so that ConcatTransform can split the values
        return data

    def __call__(self, data: dict) -> dict:
        return self.apply(data)
