import torch
import torch.nn as nn
from transformer import TransformerEncoder
from learnable_patching import Learnable1dPatching, Learnable3dPatching
from transformer import SpatioTemporalEmbeddings, TemporalEmbeddings, TransformerEncoder


class VATT(nn.Module):
    """The general Transformer for extracting different features for modalities."""

    def __init__(self,
                 # pre-transformer parameters
                 vid_temporal_patch_size=4,
                 vid_spatial_patch_size=16,
                 gaze_temporal_patch_size=4,
                 pose_temporal_patch_size=4,
                 bbox_temporal_patch_size=4,
                 # video & audio input sampling
                 random_patch_sampling=False,
                 patch_sampling_rate=0.5,
                 # transformer head parameters
                 d_model=1024,
                 d_kv=64,
                 d_ff=4096,
                 num_layers=24,
                 num_heads=16,
                 pre_norm=True,
                 use_bias=True,
                 activation="gelu",
                 dropout_rate=0.1,
                 layer_norm_epsilon=1e-6,
                 # positional embedding parameters
                 max_vid_temporal_buckets=8,
                 max_vid_spatial_buckets=14,
                 max_gaze_temporal_buckets=8,
                 max_pose_temporal_buckets=8,
                 max_bbox_temporal_buckets=8, 
                 # final head parameters
                 d_post_proj=1024,
                 post_proj_activation="gelu"):
        super(VATT, self).__init__()

        self.d_model = d_model

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            NotImplementedError 

        if post_proj_activation == "gelu":
            self.post_proj_activation = nn.GELU()
        elif post_proj_activation == "relu":
            self.post_proj_activation = nn.ReLU()
        else:
            NotImplementedError 

        # define pre-tx projection
        self.raw_to_embeddings = nn.ModuleDict({
            "video": Learnable3dPatching(out_channels=d_model,
                                         spatial_patch_size=vid_spatial_patch_size,
                                         temporal_patch_size=vid_temporal_patch_size),
            "gaze": Learnable1dPatching(in_channels=6,
                                        out_channels=d_model,
                                        temporal_patch_size=gaze_temporal_patch_size),
            "pose": Learnable1dPatching(in_channels=150,
                                        out_channels=d_model,
                                        temporal_patch_size=pose_temporal_patch_size),
            "bbox": Learnable1dPatching(in_channels=108,
                                        out_channels=d_model,
                                        temporal_patch_size=bbox_temporal_patch_size)
        })
        
        # define sampling-related params
        self.use_random_patches = random_patch_sampling
        self.patch_sampling_rate = patch_sampling_rate
        self.max_buckets = {
            "video": max_vid_temporal_buckets * (max_vid_spatial_buckets ** 2),
            "gaze": max_gaze_temporal_buckets,
            "pose": max_pose_temporal_buckets,
            "bbox": max_bbox_temporal_buckets,
        }
        self.max_num_patches = {
            "video": int(self.patch_sampling_rate * self.max_buckets["video"]),
            "gaze": int(self.patch_sampling_rate * self.max_buckets["gaze"]),
            "pose": int(self.patch_sampling_rate * self.max_buckets["pose"]),
            "bbox": int(self.patch_sampling_rate * self.max_buckets["bbox"]),
        }
        assert self.max_buckets["video"] > self.max_num_patches["video"], (
            "Max number of video positional buckets should be bigger than max"
            " number of video input patches"
            )
        assert self.max_buckets["gaze"] > self.max_num_patches["gaze"], (
            "Max number of gaze positional buckets should be bigger than max"
            " number of gaze input patches"
            )
        assert self.max_buckets["pose"] > self.max_num_patches["pose"], (
            "Max number of pose positional buckets should be bigger than max"
            " number of pose input patches"
            )
        assert self.max_buckets["bbox"] > self.max_num_patches["bbox"], (
            "Max number of bbox positional buckets should be bigger than max"
            " number of bbox input patches"
            )

        # define positional embedding module
        self.pos_embedding_lookup = nn.ModuleDict({
            "video": SpatioTemporalEmbeddings(hidden_size=self.d_model,
                                              max_temporal_buckets=max_vid_temporal_buckets,
                                              max_vertical_buckets=max_vid_spatial_buckets,
                                              max_horizontal_buckets=max_vid_spatial_buckets),
            "gaze": TemporalEmbeddings(hidden_size=self.d_model,
                                       max_temporal_buckets=max_gaze_temporal_buckets),
            "pose": TemporalEmbeddings(hidden_size=self.d_model,
                                       max_temporal_buckets=max_pose_temporal_buckets),
            "bbox": TemporalEmbeddings(hidden_size=self.d_model,
                                       max_temporal_buckets=max_bbox_temporal_buckets)
        })

        # define transformer head  
        self.tx = TransformerEncoder(d_model=d_model,
                                     d_kv=d_kv,
                                     d_ff=d_ff,
                                     num_layers=num_layers,
                                     num_heads=num_heads,
                                     pre_norm=pre_norm,
                                     use_bias=use_bias,
                                     activation=activation,
                                     dropout_rate=dropout_rate,
                                     layer_norm_epsilon=layer_norm_epsilon)

        # define post-tx projection head - it could be logits or embd space
        self.post_proj = nn.ModuleDict({
            "video": nn.Sequential(nn.Linear(d_model, d_post_proj),
                                   self.post_proj_activation),
            "gaze": nn.Sequential(nn.Linear(d_model, d_post_proj),
                                  self.post_proj_activation),
            "pose": nn.Sequential(nn.Linear(d_model, d_post_proj),
                                  self.post_proj_activation),
            "bbox": nn.Sequential(nn.Linear(d_model, d_post_proj),
                                  self.post_proj_activation)
        })

    def _flatten_inputs(self,
                        inputs):
        input_shape = inputs.shape 
        bs = inputs.shape[0]
        d_embd = inputs.shape[-1]
        inputs = inputs.view(bs, -1, d_embd)

        return inputs, input_shape

    def _append_special_tokens(self, 
                              inputs, 
                              modality):
        batch_size = inputs.shape[0]
        agg_token = {
            "video": torch.nn.Parameter(torch.Tensor(self.d_model,)),
            "gaze": torch.nn.Parameter(torch.Tensor(self.d_model,)),
            "pose": torch.nn.Parameter(torch.Tensor(self.d_model,)),
            "bbox": torch.nn.Parameter(torch.Tensor(self.d_model,)),
        }
        special_embd = agg_token[modality][None, None, :].to(inputs.device)
        special_embd = special_embd.repeat(batch_size, 1, 1)

        return torch.cat([special_embd, inputs], dim=1)

    def _random_patch_selection(self, 
                                inputs,
                                training,
                                input_shape,
                                modality):
        if training: #and modality != "text": TODO: think about this 
            # get inputs dimensions
            batch_size, seq_len, dim = inputs.shape

            # shuffle on temporal axis and gather the first max_num_patches
            temporal_idx = torch.arange(seq_len)
            temporal_idx = temporal_idx[torch.randperm(seq_len)][None, :].to(inputs.device)
            temporal_idx = temporal_idx.repeat(batch_size, 1)

            batch_idx = torch.arange(batch_size)[:, None].to(inputs.device)
            batch_idx = batch_idx.repeat(1, seq_len)

            gather_idx = torch.stack([batch_idx, temporal_idx], dim=2)

            inputs = torch.gather(inputs, 1, gather_idx)[:, :self.max_num_patches[modality], :]
            input_shape = [batch_size, self.max_num_patches[modality], dim]

        return inputs, input_shape

    def _extend_attn_mask(self, 
                          attention_mask):
        attn_mask_shape = attention_mask.shape
        if len(attn_mask_shape) > 2:
            raise NotImplementedError

        batch_size = attn_mask_shape[0]
        extention_mask = torch.ones((batch_size, 1), dtype=attention_mask.dtype) 
        extended_attention_mask = torch.cat([extention_mask, attention_mask], dim=1)
        return extended_attention_mask

    def _modality_call(self, 
                       inputs, 
                       modality, 
                       training=False,
                       attention_mask=None, 
                       input_shape=None):
        # linear projection to d_model
        embeddings = self.raw_to_embeddings[modality](inputs)

        # flatten inputs if not flattened already
        if input_shape is None:
            embeddings, input_shape = self._flatten_inputs(embeddings)
        else:
            is_flattened = len(inputs.shape) == 3
            assert is_flattened, (
                "if input_shape provided, inputs should be flattened and have rank 3")

        # add modality-specific positional encoding embeddings
        embeddings = self.pos_embedding_lookup[modality](embeddings,
                                                         input_shape)

        # randomly choose "max_num_patches" tokens
        if self.use_random_patches:
            embeddings, input_shape = self._random_patch_selection(embeddings,
                                                                   training,
                                                                   input_shape,
                                                                   modality)

        # append modalities special tokens: [vid, aud, txt]
        tx_inputs = self._append_special_tokens(embeddings, modality)

        # extend attention_mask accordingly
        if attention_mask is not None:
            attention_mask = self._extend_attn_mask(attention_mask)

        # call Transformer
        tx_outputs = self.tx(tx_inputs, attention_mask)

        # get last hidden states and perform final linear projection
        last_hidden_states = tx_outputs["hidden_states"][-1]
        modality_outputs = self.post_proj[modality](last_hidden_states)
        output_shape = list(input_shape[:-1]) + [modality_outputs.shape[-1]]

        features_pooled = modality_outputs[:, 0, :]
        features = modality_outputs[:, 1:, :].reshape(output_shape)

        # add token-level Transformer outputs
        outputs = {"features_pooled": features_pooled,
                   "features": features}

        return outputs

    def forward(self,
                inputs,
                training=False):
        outputs = {}

        for modality in ["video", "gaze", "pose", "bbox"]:
          modality_inputs = inputs[modality]["data"] 
          modality_attn_mask = inputs[modality].get("attention_mask", None)
          outputs[modality] = self._modality_call(inputs=modality_inputs,
                                                  modality=modality,
                                                  training=training,
                                                  attention_mask=modality_attn_mask)

        return outputs
    


if __name__ == "__main__":

    device = "cuda"

    bs, t = 1, 32 

    data = {
        "video": {
            "data": torch.ones(bs, t, 3, 80, 80).to(device)
        },
        "gaze": {
            "data": torch.ones(bs, t, 6).to(device)
        },
        "pose": {
            "data": torch.ones(bs, t, 150).to(device)
        }, 
        "bbox": {
            "data": torch.ones(bs, t, 108).to(device)
        }
    }

    model = VATT().to(device)

    out = model(data) 

    print('Done.')
