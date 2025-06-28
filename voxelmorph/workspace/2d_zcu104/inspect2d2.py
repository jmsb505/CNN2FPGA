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
    model = tf.keras.models.load_model("models/vxm2d_model_ez.h5",
                                       custom_objects=custom_objects,
                                       compile=False)

    inspector = VitisInspector(target="DPUCZDX8G_ISA1_B4096")

    inspector.inspect_model(
        model,
        input_shape=[
            [1, 224, 192, 1], 
            [1, 224, 192, 1]   
        ],
        dump_model=True,
        dump_model_file="outputs2d_zcu104/inspect_out/inspect_model_zcu104.h5",
        plot=True,
        plot_file="outputs2d_zcu104/inspect_out/model_inspect_zcu104.svg",
        dump_results=True,
        dump_results_file="outputs2d_zcu104/inspect_out/inspect_results_z104.txt",
        verbose=0
    )

print("Inspection complete")
