## Installation
1. Clone repository
```
git clone --recurse-submodules git@github.com:DozenDucc/VGD.git
cd VGD
```
2. Create conda environment
```
conda create -n vgd python=3.9 -y
conda activate vgd
```
3. Install our fork of DPPO 
```
cd dppo
pip install -e .
pip install -e .[robomimic]
pip install -e .[gym]
cd ..
```
4. Install our fork of Stable Baselines3
```
cd stable-baselines3
pip install -e .
cd ..
```

## Running VGD
To run VGD on Robomimic, call
```
python train_vgd.py --config-path=cfg/robomimic --config-name=vgd_can_guided.yaml
```
Replace the config file above with the desired one. 


# Introduction 

# Method 

# Experimental Results 

Baselines: 
- VGD
- DSRL



## Can Task 

<div align="center">
  <table style="border-collapse: collapse; border: none;">
    <tr>
      <td align="center" style="border: none; padding-right: 20px;">
        <b>DDIM = 4</b><br><br>
        <img src="https://github.com/user-attachments/assets/608f96b4-653a-4c54-ab05-2809439a7c65" width="400"><br>
      </td>
      <td align="center" style="border: none; padding-left: 20px;">
        <b>DDIM = 8</b><br><br>
        <img src="https://github.com/user-attachments/assets/e9229751-32b0-4e7f-8019-c1730bb4b98d" width="400"><br>
      </td>
    </tr>
  </table>
</div>

## Square Task 

<div align="center">
  <table style="border-collapse: collapse; border: none;">
    <tr>
      <td align="center" style="border: none; padding-right: 20px;">
        <b>DDIM = 4</b><br><br>
        <img src="https://github.com/user-attachments/assets/7c4d787b-2278-4683-94d5-97a3f6e8c556" width="400"><br>
      </td>
      <td align="center" style="border: none; padding-left: 20px;">
        <b>DDIM = 8</b><br><br>
        <img src="https://github.com/user-attachments/assets/145d3a87-c982-4f54-afbb-aa63ddc660be" width="400"><br>
      </td>
    </tr>
  </table>
</div>


# Analysis

## Correlation between TD error and λ<sub>t</sub>
<div align="center">
  <table style="border-collapse: collapse; border: none;">
    <tr>
      <td align="center" style="border: none; padding-right: 20px;">
        <img src="https://github.com/user-attachments/assets/edf41432-813c-4e5c-8eed-ae4113fe789d" width="400"><br>
        <sub><b>Figure #. TD Error Visualization of Dynamic VGD on the Robomimic Can Task.</b></sub>
      </td>
      <td align="center" style="border: none; padding-left: 20px;">
        <img src="https://github.com/user-attachments/assets/c3123d68-b1f6-4676-98dc-23e495a24b04" width="400"><br>
        <sub><b>Figure #. Guidance Value Visualization of Dynamic VGD on the Robomimic Can Task. </b></sub>
      </td>
    </tr>
  </table>
</div>


## Citation

If you use this code, please cite:
<pre>
@inproceedings{ye2025vgd,
  title     = {Steering Diffusion Policies with Value-Guided Denoising},
  author    = {Ye, Hanming},
  booktitle = {NeurIPS 2025 Workshop on Embodied World Models for Decision Making},
  year      = {2025}
}
</pre>

## Acknowledgements
This repository builds on [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) and [DPPO](https://github.com/irom-princeton/dppo). Parts of code adapted from [DSRL](https://github.com/ajwagen/dsrl)
