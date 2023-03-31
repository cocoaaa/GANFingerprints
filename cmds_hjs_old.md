# Create a dataset of synthetic images using ProGAN, SNGAN, CramerGAN, MMDGAN
We use the released ckpts (trained on either celeba or lsun-bedroom)
These models are trained with the following preprocessing on celeba, lsun-bedroom:
- CelebA: take the aligned,cropped face dataset (img_aligned_faces.zip)
  - Preprocessing for training the gans: crop each image centered at (x,y)=(89,121), with size 128x128

- LSUN-bedroom: use the first 200k images
  - Preprocessing for training the gans: center-crop to square size by fitting to the shorter side length, resize them to 128x128

## Steps I did (20230211-174616): 
- Download each ckpts for each gan as indicated in yu2019's readme
- Move the ckpts to each model's folder in the codebase, as indicated in the readme
- [ ] Need to sample from each ckpts to collect synthetic images from each gan ckpt
  
## Sample from Progan ckpts
Sample 100k images from each progan's ckpt (we have 1 + 10(with diff. random seed)) for each dataset (ie. {celeba, lsun-bedroom})
- [x] Generate celeba synthetic images using progan ckpts (1 + 10 ckpts)
    - cmd:
cd ProGAN/
export nsamples=10000
nohup python3 run.py \
--app gen \
--model_path models/celeba_align_png_cropped.pkl \
--out_image_dir samples/celeba_align_png_cropped/ \
--num_pngs $nsamples \
--gen_seed 0 &

    - pid:[1] 32584   -- 20230211-175546 
    - finished?: yes  -- checked at 20230211-180945 
    - output_dir: ProGAN/samples/celeba_align_png_cropped


- [ ] Generate lsun-bedroom synthetic images using progan ckpts (1 + 10 ckpts)
    - cmd:
    - pid:
    - output_dir:

## Sample from SNGAN ckpts
- [ ] Generate 100k celeba synthetic images using sngan ckpt (we have 1 ckpt)
    - cmd:
conda activate tf_gpu
mamba install chainer

# here -- > mamba install cupy -- this is giving me an error:
LOCKERROR: It looks like conda is already doing something.
The lock ['/home/hayley/miniconda3/pkgs.pid4406.conda_lock'] was found. Wait for it to finish before continuing.
If you are sure that conda is not running, remove it and try again.
You can also use: $ conda clean --lock
-- possible soln: rm the offending lock file: /home/hayley/miniconda3/pkgs.pid4406.conda_lock
(src: https://github.com/mamba-org/mamba/issues/1200#issuecomment-952784527)
20230211-182129 -- paused here

cd SNGAN/
export nsamples=10000
export nsamples=10

nohup python3 evaluations/gen_images.py \
--config_path configs/sn_projection_celeba.yml \
--snapshot models/celeba_align_png_cropped.npz \
--results_dir samples/celeba_align_png_cropped \
--num_pngs $nsamples \
--seed 0 \
&

- tr
    - pid: 
    - output_dir: SNGAN/samples

- [ ] Generate 100k lsun-bedroom synthetic images using sngan ckpt (we have 1 ckpt) 
    - cmd:
    - pid:
    - output_dir:


## Sample from CramerGAN ckpts
- [ ] Generate 100k celeba synthetic images using cramer-gan ckpt (we have 1 ckpt)
    - cmd:


# here -- paused (next item after fixing the ssngan sampling (package missing, error))
20230211-182157

conda actiavte tf_gpu

cd CramerGAN/
export nsamples=10000
python3 gan/main.py \
--is_train True \
--dataset celebA \
--data_dir ../celeba_align_png_cropped/ \
--checkpoint_dir models/ \
--sample_dir samples/ \
--no_of_samples $nsamples \
--log_dir logs/ \
--model cramer --name cramer_gan \
--architecture g_resnet5 --output_size 128 --dof_dim 256 \
--gradient_penalty 10. \
--MMD_lr_scheduler \
--random_seed 0


    - pid:
    - output_dir:

- [ ] Generate 100k lsun-bedroom synthetic images using cramer-gan ckpt (we have 1 ckpt) 
    - cmd:
    - pid:
    - output_dir:


## Sample from MMDGAN ckpts
- [ ] Generate 100k celeba synthetic images using MMDGAN ckpt (we have 1 ckpt)
    - cmd:
    - pid:
    - output_dir:

- [ ] Generate 100k lsun-bedroom synthetic images using MMDGAN ckpt (we have 1 ckpt)
    - cmd:
    - pid:
    - output_dir:
 