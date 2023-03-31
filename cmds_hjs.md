# Train yu19's enc+classifier on my GM datasets (gm64, gm256) for *multi*-class classification
1. dataset preparation
   - [ ] run the code as in readme to generate tfrecord datafiles optimized for tensorflow(?)
   - [ ] make sure when tensorflow dataset  loads those procoessed datafiles, 
         it actually loads multi-class labels properly
```bash

conda activate tf_gpu
cd classifier


in_dir=/data/datasets/neurips23_gm256/
out_dir=/docker/data/gm256_tfrecords
res=256
max_samples_per_subdir=100000


python data_preparation.py \
--in_dir $in_dir --out_dir $out_dir \
--resolution $res --max_samples_per_subdir $max_samples_per_subdir

```
- started: 20230328-202638
- pid: 2760


1. Modify classifier/networks.py to output label in a multi-class label set
   Y = {0 (real), 1 (gm_1), ... , M (gm_M)}

   - [ ] check if the loss function implemented in tensorflow as is works for multi-class labels without an error
