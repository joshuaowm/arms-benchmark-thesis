"""OSISeg training config: osiseg_base with the batch split into micro-batches.

A physical batch of 4 with 3 accumulation steps gives the base config's batch of
12 on an 11 GB GPU. Everything else comes from osiseg_base. ARMS_RES=1024 switches
the input and output size to 1024.
"""
import importlib.util as _ilu
import os as _os

# load osiseg_base.py from this folder
_v2_path = _os.path.join(_os.path.dirname(__file__), 'osiseg_base.py')
_spec = _ilu.spec_from_file_location('osiseg_base', _v2_path)
_v2 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_v2)

# re-export the base config classes unchanged
common         = _v2.common
data           = _v2.data
image_encoder  = _v2.image_encoder
prompt_encoder = _v2.prompt_encoder
mask_decoder   = _v2.mask_decoder
model          = _v2.model
test           = _v2.test


class train:
    # same as osiseg_base.train
    seed        = _v2.train.seed                # 42
    num_workers = _v2.train.num_workers
    epochs      = _v2.train.epochs              # replaced by the shared cap in train_osiseg.py
    lr          = _v2.train.lr                  # 1e-4
    out_size    = _v2.train.out_size
    debug       = _v2.train.debug
    choice_point_type = _v2.train.choice_point_type
    prompt_types = _v2.train.prompt_types       # only logged; see osiseg_base
    class_weights = _v2.train.class_weights
    # physical batch x accumulation steps = 12, as in the base config;
    # ARMS_BATCH / ARMS_ACCUM change the split (e.g. for 1024 input)
    batch_size  = int(_os.environ.get('ARMS_BATCH', 4))    # 12 at once runs out of memory on 11 GB
    grad_accum  = int(_os.environ.get('ARMS_ACCUM', 3))    # 4 x 3 = 12


# --- input resolution ----------------------------------------------------
# 512 by default; ARMS_RES=1024 sets the encoder input and the train/test output
# size together. arms/dataset_osiseg.py reads the same variable.
_R = int(_os.environ.get('ARMS_RES', 512))
if _R != 512:
    image_encoder.image_size = (_R, _R)
    train.out_size = _R
    test.out_size = _R
    # the prompt encoder's frame and embedding size were set from 512 in the base
    # config and must follow, or click coordinates land in the wrong place
    prompt_encoder.input_image_size = (_R, _R)
    _emb = _R // image_encoder.patch_size
    prompt_encoder.image_embedding_size = (_emb, _emb)
