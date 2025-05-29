import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import voxelmorph as vxm
from voxelmorph.tf.layers import VecInt, RescaleTransform, SpatialTransformer
from tensorflow_model_optimization.python.core.quantization.keras.vitis.vitis_quantize import VitisQuantizer

FLOAT_MODEL       = "/workspace/models/vxm2d_model_tf.h5"
QUANT_MODEL_OUT   = "/workspace/outputs2d/quantize_out/q_model.h5"
CALIB_DATA_DIR    = "/workspace/calibration_dataset"
CALIB_BATCH_SIZE  = 8
NUM_CALIB_SAMPLES = 20
TARGET_DPU        = "DPUCVDX8H_ISA1_F2W2_8PE"

custom_objects = {
    "VxmDense":          vxm.networks.VxmDense,
    "VecInt":            VecInt,
    "RescaleTransform":  RescaleTransform,
    "SpatialTransformer": SpatialTransformer
}

with tf.keras.utils.custom_object_scope(custom_objects):
    float_model = load_model(FLOAT_MODEL, compile=False)

def pair_generator():
    files = sorted(os.listdir(CALIB_DATA_DIR))[:NUM_CALIB_SAMPLES]
    for fn in files:
        arr = np.load(os.path.join(CALIB_DATA_DIR, fn))
        mv, fx = arr[0], arr[1]
        yield mv, fx


ds = tf.data.Dataset.from_generator(
    pair_generator,
    output_signature=
        (tf.TensorSpec((192,224,1), tf.float32), tf.TensorSpec((192,224,1), tf.float32)),
    
).batch(CALIB_BATCH_SIZE)

quantizer = VitisQuantizer(
    float_model,
    target=TARGET_DPU,
    custom_objects=custom_objects
)

ds_for_keras = ds.map(lambda mv, fx: ((mv, fx),))
quant_model = quantizer.quantize_model(calib_dataset=ds_for_keras)

os.makedirs(os.path.dirname(QUANT_MODEL_OUT), exist_ok=True)
quant_model.save(QUANT_MODEL_OUT)
print(f"Quantized model saved to {QUANT_MODEL_OUT}")
