"""OSISeg base config: SAM ViT-B with adapters in the image encoder.

Defines the encoder, prompt encoder and decoder shapes and the training
hyperparameters. The data paths and sam_checkpoint are filled in at runtime by
train/train_osiseg.py.
"""
import os

from arms.prompts import PROMPT_TYPES as _PROMPT_TYPES


class common:
    task       = 'arms_osiseg'
    output_dir = ''            # set at runtime
    gpu_id     = 0


class data:
    # all paths set at runtime from --experiment-dir (see train_osiseg.py)
    train_image_path = ''; train_ann_path = ''
    val_image_path   = ''; val_ann_path   = ''
    test_image_path  = ''; test_ann_path  = ''
    prompt_dir       = ''


class image_encoder:
    adapter      = 'sam_para_conv'
    vit_name     = 'vit_b'
    encoder_depth = 12
    encoder_embed_dim = 768
    image_size   = (512, 512)
    patch_size   = 16
    window_size  = 14
    mlp_ratio    = 4
    qkv_bias     = True
    use_rel_pos  = True
    out_channels = 256
    multi_outputs = True
    encoder_global_attn_indexes = [2, 5, 8, 11]


class prompt_encoder:
    embed_dim           = 256
    input_image_size    = image_encoder.image_size
    image_embedding_size = (
        input_image_size[0] // image_encoder.patch_size,
        input_image_size[0] // image_encoder.patch_size,
    )
    mask_in_chans = 16


class mask_decoder:
    transformer_depth         = 2
    transformer_embedding_dim = 256
    transformer_mlp_dim       = 2048
    transformer_num_heads     = 8
    num_multimask_outputs     = 3
    transformer_dim           = 256
    iou_head_depth            = 3
    iou_head_hidden_dim       = 256
    is_rein_mode              = False


class model:
    image_encoder     = image_encoder
    prompt_encoder    = prompt_encoder
    mask_decoder      = mask_decoder
    use_refine_decoder = ''


class train:
    seed        = 42
    batch_size  = 12
    num_workers = 4
    epochs      = 100
    lr          = 1e-4
    out_size    = 512
    debug       = False
    choice_point_type = 'center_point'
    # only logged: OSISeg's first prompt comes from arms.prompts_native
    prompt_types = list(_PROMPT_TYPES)
    class_weights = [1.0, 1.0, 3.0, 5.0, 6.0, 8.0]


class test:
    predict_flag   = False
    sam_checkpoint = ''        # set at runtime from the chosen backbone
    checkpoint     = ''
    out_size       = 512
    output_dir     = ''
