import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import Literal, Union, List
from torchvision import transforms as tvf

_DINO_V2_MODELS = Literal["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"]
_DINO_FACETS = Literal["query", "key", "value", "token"]


class DinoV2ExtractFeatures:

    def __init__(self, dino_model: _DINO_V2_MODELS, layer: int,
                 facet: _DINO_FACETS = "token", use_cls=False,
                 norm_descs=True, device: str = "cpu") -> None:

        self.vit_type: str = dino_model
        # self.dino_model: nn.Module = torch.hub.load('facebookresearch/dinov2', dino_model)
        self.dino_model: nn.Module = torch.hub.load('/home/ubuntu/Documents/TriSSA-main/facebookresearch_dinov2_main', 'dinov2_vitg14', source='local')
        self.device = torch.device(device)
        self.dino_model = self.dino_model.eval().to(self.device)
        self.layer: int = layer
        self.facet = facet
        if self.facet == "token":
            self.fh_handle = self.dino_model.blocks[self.layer]. \
                register_forward_hook(
                self._generate_forward_hook())
        else:
            self.fh_handle = self.dino_model.blocks[self.layer]. \
                attn.qkv.register_forward_hook(
                self._generate_forward_hook())
        self.use_cls = use_cls
        self.norm_descs = norm_descs
        # Hook data
        self._hook_out = None

    def _generate_forward_hook(self):
        def _forward_hook(module, inputs, output):
            self._hook_out = output

        return _forward_hook

    def __call__(self, img: torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            res = self.dino_model(img)
            if self.use_cls:
                res = self._hook_out
            else:
                res = self._hook_out[:, 1:, ...]
            if self.facet in ["query", "key", "value"]:
                d_len = res.shape[2] // 3
                if self.facet == "query":
                    res = res[:, :, :d_len]
                elif self.facet == "key":
                    res = res[:, :, d_len:2 * d_len]
                else:
                    res = res[:, :, 2 * d_len:]
        if self.norm_descs:
            res = F.normalize(res, dim=-1)
        self._hook_out = None  # Reset the hook
        return res

    # def __del__(self):
    #     self.fh_handle.remove()


if __name__ == "__main__":
    desc_layer: int = 31
    desc_facet: Literal["query", "key", "value", "token"] = "value"
    device = torch.device("cuda")

    extractor = DinoV2ExtractFeatures("dinov2_vitg14", desc_layer, desc_facet, device=device)

