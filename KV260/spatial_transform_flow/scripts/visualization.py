
import numpy as np
import tensorflow as tf
import voxelmorph as vxm
import matplotlib.pyplot as plt

n_samples=20
mse_list, grad_list, loss_list =[], [], []

def getResults(i):
    moving_b = np.load(f'moving/moving{i}.npy').astype(np.float32)
    fixed_b = np.load(f'fixed/fixed{i}.npy').astype(np.float32)
    pad = np.load(f'pads/pad{i}.npy').astype(np.float32)

    mov_img = moving_b[0,...,0]
    fix_img = fixed_b[0,...,0]

    flow_all = pad[..., :2]
    flow_pred = flow_all[0]

    inp_mov = tf.keras.Input(shape=moving_b.shape[1:], name='mov')
    inp_flow = tf.keras.Input(shape=flow_pred.shape, name='flow')
    warp_out = vxm.layers.SpatialTransformer(
        interp_method='linear',
        indexing='ij'
    )([inp_mov, inp_flow])
    st_model = tf.keras.Model([inp_mov, inp_flow], warp_out)

    warp_pred = st_model.predict([moving_b, flow_all], verbose=0)
    flow = flow_pred
    warped = warp_pred[0,...,0]

    mse = np.mean((warped - fix_img)**2)
    dx = flow[1:,:,:] - flow[:-1,:,:]
    dy = flow[:,1:,:] - flow[:,:-1,:]
    grad_loss = (np.sum(dx**2) + np.sum(dy**2)) / np.prod(flow.shape)
    total_loss = mse + 0.01 * grad_loss

    mse_list.append(mse)
    grad_list.append(grad_loss)
    loss_list.append(total_loss)

    return([flow, mov_img, fix_img, warped])

def visualization(flow, mov_img, fix_img, warped):
    mean_mse = np.mean(mse_list)
    std_mse = np.std(mse_list)
    mean_grad = np.mean(grad_list)
    std_grad = np.std(grad_list)
    mean_loss = np.mean(loss_list)
    std_loss = np.std(loss_list)

    print(f'avg mse: {mean_mse:.6e} \u00B1 {std_mse:.6e}')
    print(f'avg grad: {mean_grad:.6e} \u00B1 {std_grad:.6e}')
    print(f'avg loss: {mean_loss:.6e} \u00B1 {std_loss:.6e}')
    h, w = flow.shape[:2]
    step = 8
    y = np.arange(0, h, step)
    x = np.arange(0, w, step)
    X, Y = np.meshgrid(x, y)
    U = flow[Y, X, 1]
    V = flow[Y, X, 0]

    plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.imshow(mov_img, cmap='gray'); plt.axis('off')
    plt.title('Moving')

    plt.subplot(132)
    plt.imshow(fix_img, cmap='gray'); plt.axis('off')
    plt.title('Fixed')

    plt.subplot(133)
    plt.imshow(warped, cmap='gray')
    plt.title('Warped Image + Flow')


    plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1,
           width=0.003, headwidth=3, headlength=4, alpha=0.8, color='r')
    plt.axis('off'); plt.tight_layout(); plt.show()


def main():
    last_result = []
    for i in range (20):
        if i == 0:
            first_result = getResults(i)
        else: 
            last_result = getResults(i)
    visualization(*first_result)

if __name__ == '__main__':
    main()
