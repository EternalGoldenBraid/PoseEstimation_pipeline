**Undergraduate thesis project.**
# Setup
Please start by installing [mamba](https://github.com/conda-forge/miniforge#mambaforge), [Miniconda3](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html) or conda
with Python3.9 or above.

either run `<mamba/conda> install -f environment.yml` or install dependencies manually:

<details >
<summary> Manual dependency installation </summary>
Instal the following dependencies (Conda/Mamba or pip):

- [Pytorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
- numpy, opencv, trimesh, pyrender, scikit-image

At the time of writing, pip only. (If using realsense camera for RGBD)
- [Pyrealsense](https://pypi.org/project/pyrealsense/)
	- `pip install pyrealsense2==2.50.0.3812`
</details>
<br>

### Download Model weights for OVE6D
`mkdir checkpoints; cd checkpoints`

**Pose estimation weights** \
- `wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1aXkYOpvka5VAPYUYuHaCMp0nIvzxhW9X' -O OVE6D_pose_model.pth` \
or
- `wget https://drive.proton.me/urls/2GQBGB2DH4#aLLLp43rOm8M -O OVE6D_pose_model.pth`

or manually from:
OVE6D: [Project page](https://dingdingcai.github.io/ove6d-pose/) 
	- https://drive.google.com/drive/folders/16f2xOjQszVY4aC-oVboAD-Z40Aajoc1s?usp=sharing).
	

# To experiment with custom objects
- Provide 3D model of the query object in *.ply format in `Dataspace/<dataset_name>/models_eval/`
	- Dataset name defined in `configs/config.py` as `DATASET_NAME`.
	- Provide `name` and `diameter` of 3D model in `Dataspace/<dataset_name>/models_eval/models_info.json`.
	- Adjust `MODEL_SCALING` in `config.py` to whatever scale (meters/mm) you're using for the 
	3D models.

- Attach a realsense camera or implement a camera module for camera of choice 
	as in `cam_control.py` at `utils/cam_control.py`
	
- Setup a greenscreen and adjust the chroma keying parameters in the pop up window at runtime.
	- Alternatively use a segmentator of choice that provides a binary mask as in 
		`utility\load_segmentation_model_chroma.py`.


# Some qualitative results
https://user-images.githubusercontent.com/51406001/208401627-bbaafb13-d04a-4c0f-979c-d899aa943b74.mp4


https://user-images.githubusercontent.com/51406001/208401652-a1acfd67-91e3-486c-98a5-f246c6e0b97e.mp4



# Acknowledgements
- OVE6D: [Project page](https://dingdingcai.github.io/ove6d-pose/) 
