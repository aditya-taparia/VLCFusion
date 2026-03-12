"""
MultimodalDetr variant that uses a frozen MetaCLIP-2 image encoder to produce
conditioning vectors on-the-fly, instead of precalculated condition files.

The visible (RGB) image is fed through MetaCLIP's vision tower to obtain an
embedding which conditions the CrossCBAMDiTFusionV2 blocks.

Supported MetaCLIP-2 variants (default: L14, good speed/quality tradeoff):
  - facebook/metaclip-2-worldwide-s16       (ViT-S/16, 384-dim, ~22M params, fastest)
  - facebook/metaclip-2-worldwide-l14       (ViT-L/14, 768-dim, ~304M params)
  - facebook/metaclip-2-worldwide-huge-quickgelu (ViT-H/14, 1024-dim, ~632M params)
"""

import torch
import torch.nn as nn
from transformers.models.detr.modeling_detr import (
    DetrObjectDetectionOutput,
    DetrModelOutput,
)
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForObjectDetection,
)
from typing import Optional, Tuple, Union

# from cross_cbam_dit_v2_utils import (
#     CrossCBAMDiTTransformLayerV2,
#     CrossCBAMDiTTransformQueriesV2,
# )

from cross_cbam_dit_v5_utils import CrossCBAMDiTTransformLayerV5, CrossCBAMDiTTransformQueriesV5


class MetaCLIPConditioner(nn.Module):
    """Frozen MetaCLIP-2 vision encoder that maps an RGB image to a
    conditioning vector of shape [B, projection_dim].

    Runs in bfloat16 with SDPA attention for speed. The output is cast
    back to float32 so downstream fusion layers stay in full precision.
    """

    def __init__(
        self,
        model_name: str = "facebook/metaclip-2-worldwide-l14",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.dtype = torch_dtype
        clip_model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation="sdpa",
        )
        self.vision_model = clip_model.vision_model
        self.visual_projection = clip_model.visual_projection
        self.projection_dim: int = clip_model.config.projection_dim

        self.vision_model.requires_grad_(False)
        self.visual_projection.requires_grad_(False)
        self.vision_model.eval()
        self.visual_projection.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.vision_model.eval()
        self.visual_projection.eval()
        return self

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, 3, H, W] RGB images preprocessed by MetaCLIP processor.
        Returns:
            [B, projection_dim] conditioning vector in float32.
        """
        pv = pixel_values.to(dtype=self.dtype)
        vision_out = self.vision_model(pixel_values=pv)
        pooled = vision_out.pooler_output
        projected = self.visual_projection(pooled)
        return projected.float()


class MultimodalDetrMetaCLIP(nn.Module):
    """MultimodalDetr with a live MetaCLIP-2 vision encoder producing
    conditioning vectors for CrossCBAMDiTFusionV2."""

    def __init__(
        self,
        model_name_1: str,
        model_name_2: str,
        config: AutoConfig,
        clip_model_name: str = "facebook/metaclip-2-worldwide-l14",
        clip_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.config = config

        # --- MetaCLIP conditioner (frozen, bf16 + SDPA) ---
        self.conditioner = MetaCLIPConditioner(
            model_name=clip_model_name,
            torch_dtype=clip_dtype,
        )
        cond_dim = self.conditioner.projection_dim

        # --- DETR backbones (frozen) ---
        model_ir = AutoModelForObjectDetection.from_pretrained(
            model_name_1, config=self.config,
            ignore_mismatched_sizes=True, low_cpu_mem_usage=False,
        )
        model_rgb = AutoModelForObjectDetection.from_pretrained(
            model_name_2, config=self.config,
            ignore_mismatched_sizes=True, low_cpu_mem_usage=False,
        )

        self.backbone_ir = model_ir.model.backbone
        self.input_projection_ir = model_ir.model.input_projection
        for m in [self.backbone_ir, self.input_projection_ir]:
            for p in m.parameters():
                p.requires_grad = False

        self.backbone_rgb = model_rgb.model.backbone
        self.input_projection_rgb = model_rgb.model.input_projection
        for m in [self.backbone_rgb, self.input_projection_rgb]:
            for p in m.parameters():
                p.requires_grad = False

        # --- Shared DETR components ---
        self.query_position_embeddings = model_ir.model.query_position_embeddings
        self.encoder = model_ir.model.encoder
        self.decoder = model_ir.model.decoder
        self.class_labels_classifier = model_ir.class_labels_classifier
        self.bbox_predictor = model_ir.bbox_predictor
        self.loss_function = model_ir.loss_function
        self.device_attr = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Fusion layers (trainable, conditioned by MetaCLIP) ---
        in_channels = 512
        out_channels = 256
        r = 2

        self.transform_layer = CrossCBAMDiTTransformLayerV5(
            in_channels=in_channels,
            out_channels=out_channels,
            cond_dim=cond_dim,
            r=r,
        )
        self.transform_queries = CrossCBAMDiTTransformLayerV5(
            in_channels=in_channels,
            out_channels=out_channels,
            cond_dim=cond_dim,
            r=r,
        )

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        clip_pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        labels: Optional[list] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], DetrModelOutput]:
        """
        Args:
            pixel_values: [B, 6, H, W] — concatenated IR (ch 0-2) + RGB (ch 3-5),
                          normalised for DETR backbones.
            clip_pixel_values: [B, 3, H_clip, W_clip] — RGB images preprocessed
                               for the MetaCLIP vision encoder (224x224).
            labels: list of dicts with detection targets.
        """
        # --- Conditioning from MetaCLIP ---
        conditions = self.conditioner(clip_pixel_values)  # [B, 1024]

        # --- Split IR / RGB ---
        ir_pixel_values = pixel_values[:, :3, ...]
        rgb_pixel_values = pixel_values[:, 3:, ...]

        batch_size, _, height, width = pixel_values.shape
        device = pixel_values.device

        if pixel_mask is None:
            pixel_mask = torch.ones((batch_size, height, width), device=device)

        # --- Backbone feature extraction ---
        features_ir, object_queries_list_ir = self.backbone_ir(ir_pixel_values, pixel_mask)
        feature_map_ir, mask_ir = features_ir[-1]
        if mask_ir is None:
            raise ValueError("IR Backbone does not return downsampled pixel mask")
        projected_feature_map_ir = self.input_projection_ir(feature_map_ir)

        features_rgb, object_queries_list_rgb = self.backbone_rgb(rgb_pixel_values, pixel_mask)
        feature_map_rgb, mask_rgb = features_rgb[-1]
        if mask_rgb is None:
            raise ValueError("RGB Backbone does not return downsampled pixel mask")
        projected_feature_map_rgb = self.input_projection_rgb(feature_map_rgb)

        # --- Fusion (conditioned by MetaCLIP embedding) ---
        feature_map = torch.cat((projected_feature_map_ir, projected_feature_map_rgb), dim=1)
        feature_map = self.transform_layer(feature_map, conditions)

        object_queries_ir = object_queries_list_ir[-1]
        object_queries_rgb = object_queries_list_rgb[-1]
        object_queries = torch.cat((object_queries_ir, object_queries_rgb), dim=1)
        object_queries = self.transform_queries(object_queries, conditions)

        mask = mask_ir & mask_rgb

        # --- Flatten and pass through encoder/decoder ---
        flattened_features = feature_map.flatten(2).permute(0, 2, 1)
        object_queries = object_queries.flatten(2).permute(0, 2, 1)
        flattened_mask = mask.flatten(1)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                inputs_embeds=flattened_features,
                attention_mask=flattened_mask,
                object_queries=object_queries,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, DetrModelOutput):
            encoder_outputs = DetrModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        query_position_embeddings = self.query_position_embeddings.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        queries = torch.zeros_like(query_position_embeddings)

        decoder_outputs = self.decoder(
            inputs_embeds=queries,
            attention_mask=None,
            object_queries=object_queries,
            query_position_embeddings=query_position_embeddings,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=flattened_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            outputs = decoder_outputs + encoder_outputs
        else:
            outputs = DetrModelOutput(
                last_hidden_state=decoder_outputs.last_hidden_state,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
                intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
            )

        sequence_output = outputs[0]
        logits = self.class_labels_classifier(sequence_output)
        pred_boxes = self.bbox_predictor(sequence_output).sigmoid()

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            outputs_class, outputs_coord = None, None
            if self.config.auxiliary_loss:
                intermediate = outputs.intermediate_hidden_states if return_dict else outputs[4]
                outputs_class = self.class_labels_classifier(intermediate)
                outputs_coord = self.bbox_predictor(intermediate).sigmoid()
            loss, loss_dict, auxiliary_outputs = self.loss_function(
                logits, labels, device, pred_boxes, self.config, outputs_class, outputs_coord,
            )

        if not return_dict:
            if auxiliary_outputs is not None:
                output = (logits, pred_boxes) + auxiliary_outputs + outputs
            else:
                output = (logits, pred_boxes) + outputs
            return ((loss, loss_dict) + output) if loss is not None else output

        return DetrObjectDetectionOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            auxiliary_outputs=auxiliary_outputs,
            last_hidden_state=outputs.last_hidden_state,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
