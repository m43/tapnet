# Reproduce Tapnet Results

## Environment Setup

First, install [OpenEXR](https://openexr.com/en/latest/). Since I do not
have admin access on my machine, I will build OpenEXR from source and
install it to `~/.local`.
```bash
cd /scratch/izar/rajic/eth-master-thesis/03-code/
git clone https://github.com/AcademySoftwareFoundation/openexr
cd openexr
# git checkout fd9d1ff55f340152ff3764617a2caef796142fc2
git checkout 8bc3741131db146ad08a5b83af9e6e48f0e94a03
mkdir build
cd build
cmake .. --install-prefix $HOME/.local
cmake --build . --target install --config Release -j 40

# Inspect the content of `$HOME/.local` to see what happened, if unsure
# Update your `.bashrc` file, e.g., add the exports below. I am not sure if all are needed.
#    export PATH="$HOME/.local/bin/:$PATH"
#    export CPATH="$HOME/.local/include/OpenEXR:$CPATH"
#    export CPATH="$HOME/.local/include/Imath:$CPATH"
#    export LIBRARY_PATH="$HOME/.local/lib:$LIBRARY_PATH"
#    export LIBRARY_PATH="$HOME/.local/lib64:$LIBRARY_PATH"
#    export LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH"
#    export LD_LIBRARY_PATH="$HOME/.local/lib64:$LD_LIBRARY_PATH"
```

You can verify that your `openexr` setup is correct by trying to install
the Python bindings below with `pip install openexr`, or by trying to
build a dummy hello world cpp program from here:
https://github.com/AcademySoftwareFoundation/openexr

Now, setup a conda environment to use for tapnet:
```bash
cd /scratch/izar/rajic/eth-master-thesis/03-code/tapnet/
conda create --name tapnet python=3.9 -y
conda activate tapnet
```

If you want to use CUDA, make sure you install the drivers and a version
of JAX that's compatible with your CUDA and CUDNN versions. Refer to the
[jax manual](https://github.com/google/jax#pip-installation-gpu-cuda) to
install JAX version with CUDA. For example:
```bash
conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

Install tensorflow and make sure it works for your setup:
```bash
pip install --upgrade pip
pip install tensorflow==2.8.2
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000]))); print('TF w/ CPU set up correctly')"
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU')); print('TF w/ GPU set up correctly')"
```

Check that you can install the Python bindings for OpenEXR, therby
verifying that OpenEXR has been set up correctly:
```bash
pip install openexr # If this fails, openexr is probably not set up correctly
```

Install the remaining requirements:
```bash
pip install -r requirements.txt
```

Make sure that the `kubric` submodule is initialized and updated:
```bash
git submodule init
git submodule update
```

Finally, make sure to add the parent directory of tapnet to
`PYTHONPATH`. This must be done again each time the shell is
reinitialized (e.g., if you open a new terminal). For convenience, you
might want to create an initialization script.
```bash
export PYTHONPATH=/scratch/izar/rajic/eth-master-thesis/03-code/:$PYTHONPATH
```

## Prepare the Data

I will download the necessary data into `./datasets`, that I soft linked
to a data folder using `ln -s /scratch/izar/rajic/eth-master-thesis/00-data datasets`.

Download TAP-Vid-DAVIS and TAP-Vid-RGB-Stacking:
```bash
cd datasets
wget https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip
wget https://storage.googleapis.com/dm-tapnet/tapvid_rgb_stacking.zip
unzip tapvid_davis.zip
unzip tapvid_rgb_stacking.zip
```

Remember to update the experiment configuration with the paths to the
downloaded datasets. You can tweak the `configs/tapnet_config.py`
configuration file directly, or you can override the config on the
command line when running the experiment, for example like this:
```bash
--config.experiment_kwargs.config.davis_points_path=./datasets/tapvid_davis/tapvid_davis.pkl
--config.experiment_kwargs.config.jhmdb_path=#TODO
--config.experiment_kwargs.config.robotics_points_path=./datasets/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl
```

## Reproduction

```bash
python ./experiment.py \
  --config ./configs/tapnet_config.py \
  --config.experiment_kwargs.config.davis_points_path=./datasets/tapvid_davis/tapvid_davis.pkl \
  --config.experiment_kwargs.config.robotics_points_path=./datasets/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl

python3 ./tapnet/experiment.py \
  --config=./tapnet/configs/tapnet_config.py \
  --jaxline_mode=eval_davis_points \
  --config.checkpoint_dir=./tapnet/checkpoint/ \
  --config.experiment_kwargs.config.davis_points_path=/path/to/tapvid_davis.pkl

python3 ./tapnet/experiment.py \
  --config=./tapnet/configs/tapnet_config.py \
  --jaxline_mode=eval_inference \
  --config.checkpoint_dir=./tapnet/checkpoint/ \
  --config.experiment_kwargs.config.inference.input_video_path=horsejump-high.mp4 \
  --config.experiment_kwargs.config.inference.output_video_path=result.mp4 \
  --config.experiment_kwargs.config.inference.resize_height=256 \
  --config.experiment_kwargs.config.inference.resize_width=256 \
  --config.experiment_kwargs.config.inference.num_points=20
```

Download a baseline checkpoint
```bash
mkdir -p checkpoint
wget --directory-prefix=checkpoint https://storage.googleapis.com/dm-tapnet/checkpoint.npy
```

## Plot Precision Plots

`TODO`

## Plot Dataset Annotations

`TODO`

## Identifying Failure Cases

`TODO`
