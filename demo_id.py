import numpy as np
from tqdm import tqdm
from network import IDreveal
import matplotlib.pyplot as plt
import argparse


def extract_embedding(x):

    device = 'cpu'  # in ('cpu', 'cuda:0', 'cuda:1')
    time = 100  # length of sequences in frames

    if isinstance(x, str):
        x = np.load(x)  # load 3ddfa features

    # insert NAN in the temporal positions where the face is not detected
    ts = int(np.nanmin(x['image_inds']))
    te = int(np.nanmax(x['image_inds']) + 1)
    inp = np.full((te - ts, x['tddfa'].shape[1]), np.nan, dtype=np.float32)
    for i, d in zip(x['image_inds'], x['tddfa']):
        if np.isfinite(i):
            inp[int(i) - ts] = d
    net = IDreveal(time=time, device=device, weights_file='./model.th')
    y = net(inp)  # apply Temporal ID Network
    y = y[np.all(np.isfinite(y), -1)]  # remove NAN positions
    return y


def main(args):

    threshold = 1.1 ** 0.5  # 1.0488

    ref_vid = f'{args.ref}'  # Reference Video

    test_vid = f'{args.test}'  #Test Video

    # extract embedded vectors for reference video
    print('Extracting embedded vectors for reference video', flush=True)
    ref_embs = np.concatenate([extract_embedding(ref_vid)], 0)
    print('Extracting embedded vectors and distance computation for test video', flush=True)
    test_embs = extract_embedding(test_vid)  # extract embedded vectors for a test video
    dist = np.min(np.min(np.sum(np.square(ref_embs[None,:,:]-test_embs[:,None,:]),-1),-1))  # compute distances

    # compare dist with threshold
    if dist < threshold:
        print('dist =', dist)
        print('The test video is real', flush=True)
    else:
        print('dist =', dist)
        print('The test video is fake', flush=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The demo of ID-Reveal')
    parser.add_argument('-ref', type=str, default='examples/results/param/ref.npz')
    parser.add_argument('-test', type=str, default='examples/results/param/test.npz')
    parser.add_argument('-m', '--mode', default='gpu', type=str, help='gpu or cpu mode')

    args = parser.parse_args()
    main(args)
