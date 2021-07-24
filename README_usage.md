# ID_Reveal
## Usage
1、Clone this repo

2、Build the cython version of NMS, Sim3DR, and the faster mesh render

`
sh ./build.sh
`

3、Run 3DDFA demos

1）running on videos

`
python demo_video.py -f examples/inputs/videos/ref.avi
`

2）running on videos smoothly by looking ahead by `n_next` frames

`
python demo_video_smooth.py -f examples/inputs/videos/ref.avi
`

4、Run IDReveal demos

`
python demo_id.py -ref examples/results/param/ref.npz -test examples/results/param/test.npz
`
