import tensorflow as tf
import voxelmorph as vxm
from voxelmorph.tf.layers import VecInt, RescaleTransform, SpatialTransformer
from tensorflow_model_optimization.python.core.quantization.keras.vitis.vitis_inspect import VitisInspector

custom_objects = {
    "VxmDense": vxm.networks.VxmDense,
    "VecInt": VecInt,
    "RescaleTransform": RescaleTransform,
    "SpatialTransformer": SpatialTransformer
}

with tf.keras.utils.custom_object_scope(custom_objects):
    model = tf.keras.models.load_model("models/vxm_dense_brain_T1_3D_mse.h5",
                                       custom_objects=custom_objects,
                                       compile=False)

    inspector = VitisInspector(target="DPUCVDX8H_ISA1_F2W2_8PE")

    inspector.inspect_model(
        model,
        input_shape=[
            [1, 160, 192, 224, 1],
            [1, 160, 192, 224, 1],
        ],
        dump_model=True,
        dump_model_file="outputs3d/inspect_out/inspect_model.h5",
        plot=True,
        plot_file="outputs3d/inspect_out/model_inspect.svg",
        dump_results=True,
        dump_results_file="outputs3d/inspect_out/inspect_results.txt",
        verbose=0
    )

print("Inspection complete")
