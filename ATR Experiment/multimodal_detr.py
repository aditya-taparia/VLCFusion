import torch, torch.nn as nn
import torch.nn.functional as F
from transformers.models.detr.modeling_detr import (
    DetrForObjectDetection,
    DetrObjectDetectionOutput,
    DetrModelOutput,
)

import transformers
from transformers import (
    AutoConfig,
    AutoModelForObjectDetection,
)
from typing import List, Optional, Tuple, Union

class MultimodalDetr(nn.Module):
    def __init__(self, model_name_1: str, model_name_2: str, config: AutoConfig, ensemble_method: str = "CBAM", n_conditions: int = 14):
        super().__init__()
        
        if ensemble_method not in ["CBAM", "FusionSSD", "CBAM_FiLM", "FusionSSD_FiLM", "FusionSSD_SelfAttention", "LearnableAlign"]:
            raise NotImplementedError(f"Ensemble method {ensemble_method} not implemented")
        self.ensemble_method = ensemble_method

        self.config = config

        model_ir  = AutoModelForObjectDetection.from_pretrained(model_name_1, config=self.config, ignore_mismatched_sizes=True)
        model_rgb = AutoModelForObjectDetection.from_pretrained(model_name_2, config=self.config, ignore_mismatched_sizes=True)
        
        # Get components from the models
        self.backbone_ir = model_ir.model.backbone
        self.input_projection_ir = model_ir.model.input_projection
        
        for module in [
            self.backbone_ir,
            self.input_projection_ir
        ]:
            for param in module.parameters():
                param.requires_grad = False
        
        # Get components from the models
        self.backbone_rgb = model_rgb.model.backbone
        self.input_projection_rgb = model_rgb.model.input_projection
        
        for module in [
            self.backbone_rgb,
            self.input_projection_rgb
        ]:
            for param in module.parameters():
                param.requires_grad = False
        
        # Common components
        self.query_position_embeddings = model_ir.model.query_position_embeddings
        self.encoder = model_ir.model.encoder
        self.decoder = model_ir.model.decoder
        self.class_labels_classifier = model_ir.class_labels_classifier
        self.bbox_predictor = model_ir.bbox_predictor
        self.loss_function = model_ir.loss_function
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.ensemble_method == "FusionSSD":
            # Convoulutional layer to project the concatenated feature maps
            self.transform_layer = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )
            
            # Convolutional layer to project the concatenated object queries
            self.transform_queries = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )
        
        elif self.ensemble_method == "FusionSSD_FiLM": 
            in_channels = 512
            out_channels = 256
            cond_dim = n_conditions
            
            from multimodal_detr_utils import FusionSSDFiLMTransformLayer, FusionSSDFiLMTransformQueries
            
            self.transform_layer = FusionSSDFiLMTransformLayer(in_channels=in_channels, out_channels=out_channels, cond_dim=cond_dim)
            self.transform_queries = FusionSSDFiLMTransformQueries(in_channels=in_channels, out_channels=out_channels, cond_dim=cond_dim)
            
        
        elif self.ensemble_method == "CBAM":
            from multimodal_detr_utils import CBAM
            
            in_channels = 512
            r = 2
            
            self.transform_layer = nn.Sequential(
                CBAM(channels=in_channels, r=r),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(512, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )
            
            # Convolutional layer to project the concatenated object queries
            self.transform_queries = nn.Sequential(
                CBAM(channels=in_channels, r=r),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(512, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )
        
        elif self.ensemble_method == "CBAM_FiLM":
            from multimodal_detr_utils import CBAMFiLMTransformLayer, CBAMFiLMTransformQueries
            
            in_channels = 512
            out_channels = 256
            r = 2
            cond_dim = n_conditions
            
            self.transform_layer = CBAMFiLMTransformLayer(in_channels=in_channels, out_channels=out_channels, r=r, cond_dim=cond_dim)
            self.transform_queries = CBAMFiLMTransformQueries(in_channels=in_channels, out_channels=out_channels , r=r, cond_dim=cond_dim)
        
        elif self.ensemble_method == "FusionSSD_SelfAttention":
            from multimodal_detr_utils import FusionSSDSelfAttentionTransformLayer, FusionSSDSelfAttentionTransformQueries
            
            in_channels = 512
            out_channels = 256
            cond_dim = n_conditions
            
            self.transform_layer = FusionSSDSelfAttentionTransformLayer(in_channels=in_channels, out_channels=out_channels)
            self.transform_queries = FusionSSDSelfAttentionTransformQueries(in_channels=in_channels, out_channels=out_channels)
        
        elif self.ensemble_method == "LearnableAlign":
            from multimodal_detr_utils import LearnableAlignTransformLayer, LearnableAlignTransformQueries
            
            in_channels = 512
            out_channels = 256
            cond_dim = n_conditions
            
            self.transform_layer = LearnableAlignTransformLayer(in_channels=in_channels, out_channels=out_channels)
            self.transform_queries = LearnableAlignTransformQueries(in_channels=in_channels, out_channels=out_channels)
        
    def forward( 
        self,
        pixel_values:torch.FloatTensor, 
        pixel_mask:torch.LongTensor = None,
        labels:list | None = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], DetrModelOutput]:
        
        # Get image_ids
        if "FiLM" in self.ensemble_method.split("_"):
            conditions = []
            for label in labels:
                conditions.append(label["conditions"])
            # print("Conditions:", conditions)

            conditions_tensor_list = []
            for condition in conditions:
                condition_tensor = torch.tensor(condition, dtype=torch.float, device=self.device)
                conditions_tensor_list.append(condition_tensor)
            conditions_tensor = torch.stack(conditions_tensor_list, dim=0)
        

        ir_pixel_values  = pixel_values[:, :3, ...]   # first three channels
        rgb_pixel_values = pixel_values[:, 3:, ...]   # last  three channels
        
        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device

        if pixel_mask is None:
            pixel_mask = torch.ones(((batch_size, height, width)), device=device)
        
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
        
        # Concatenate the feature maps
        feature_map = torch.cat((projected_feature_map_ir, projected_feature_map_rgb), dim=1)
        if "FiLM" in self.ensemble_method.split("_"):
            feature_map = self.transform_layer(feature_map, conditions_tensor)
        else:
            feature_map = self.transform_layer(feature_map)
        
        # Concatenate the object queries
        object_queries_ir = object_queries_list_ir[-1]
        object_queries_rgb = object_queries_list_rgb[-1]
        object_queries = torch.cat((object_queries_ir, object_queries_rgb), dim=1)
        if "FiLM" in self.ensemble_method.split("_"):
            object_queries = self.transform_queries(object_queries, conditions_tensor)
        else:
            object_queries = self.transform_queries(object_queries)
        
        # And operation to get the mask
        mask = mask_ir & mask_rgb
        
        # Third, flatten the feature map + position embeddings of shape NxCxHxW to NxCxHW, and permute it to NxHWxC
        # In other words, turn their shape into (batch_size, sequence_length, hidden_size)
        flattened_features = feature_map.flatten(2).permute(0, 2, 1)
        object_queries = object_queries.flatten(2).permute(0, 2, 1)
        
        flattened_mask = mask.flatten(1)
        
        # Fourth, sent flattened_features + flattened_mask + position embeddings through encoder
        # flattened_features is a Tensor of shape (batch_size, heigth*width, hidden_size)
        # flattened_mask is a Tensor of shape (batch_size, heigth*width)
        
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
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, DetrModelOutput):
            encoder_outputs = DetrModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        
        # Fifth, sent query embeddings + object_queries through the decoder (which is conditioned on the encoder output)
        query_position_embeddings = self.query_position_embeddings.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        queries = torch.zeros_like(query_position_embeddings)
        
        # decoder outputs consists of (dec_features, dec_hidden, dec_attn)
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
        
        # class logits + predicted bounding boxes
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
                logits, labels, self.device, pred_boxes, self.config, outputs_class, outputs_coord
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