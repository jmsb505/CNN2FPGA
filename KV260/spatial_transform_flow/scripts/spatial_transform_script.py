import numpy as np
import tensorflow as tf
import voxelmorph as vxm
import sys

def main():
    a = np.load('pad.npy')
    b = np.load('moving.npy')
    print(f"pad.npy  shape = {a.shape}")
    print(f"moving.npy shape = {b.shape}")

    if a.shape[-1] == 16 and b.shape[-1] == 1:
        pad, img = a, b
    elif b.shape[-1] == 16 and a.shape[-1] == 1:
        pad, img = b, a
    else:
        print("Error: neither pad.npy nor moving.npy", file=sys.stderr)
        sys.exit(1)

    print(f"Detected pad   shape = {pad.shape}")
    print(f"Detected image shape = {img.shape}")

    flow = pad[..., :2]
    print(f"Extracted flow shape = {flow.shape}")
    np.save('flow.npy', flow)

    tf_img  = tf.convert_to_tensor(img,  dtype=tf.float32)
    tf_flow = tf.convert_to_tensor(flow, dtype=tf.float32)

    st = vxm.layers.SpatialTransformer(
        interp_method='linear',
        indexing='ij'
    )

    warped_tf = st([tf_img, tf_flow])
    warped    = warped_tf.numpy()
    print(f"warped output shape = {warped.shape}")

    np.save('warped.npy', warped)
    print("Done. Written: flow.npy, warped.npy")

if __name__ == '__main__':
    main()
