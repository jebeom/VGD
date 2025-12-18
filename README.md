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

Pipeline: Pretrain Diffusion Policy → RL Fine-tune DP with DSRL/VGD/Dynamic VGD

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.

Baselines: 
- VGD: Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
- DSRL: Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.

Tasks 
- Expert demos: ??? 


<div align="center">
  <table style="border-collapse: collapse; border: none;">
    <tr>
      <td align="center" style="border: none; padding-right: 20px;">
        <b>Can</b><br>
        <img width="150" src="https://github.com/user-attachments/assets/a558a7d5-6b8b-4fc8-a953-82c90589367f">
      </td>
      <td align="center" style="border: none; padding-left: 20px;">
        <b>Square</b><br>
        <img width="150" src="https://github.com/user-attachments/assets/554cc6a6-bdd6-4ff9-952c-13b5aa895ea4">
      </td>
    </tr>
  </table>
</div>


## Can Task 

<div align="center">
  <table style="border-collapse: collapse; border: none;">
    <tr>
      <td align="center" style="border: none; padding-right: 20px;">
        <b>DDIM = 4</b><br>
        <img src="https://github.com/user-attachments/assets/608f96b4-653a-4c54-ab05-2809439a7c65" width="300"><br>
      </td>
      <td align="center" style="border: none; padding-left: 20px;">
        <b>DDIM = 8</b><br>
        <img src="https://github.com/user-attachments/assets/e9229751-32b0-4e7f-8019-c1730bb4b98d" width="300"><br>
      </td>
    </tr>
  </table>
</div>

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.

## Square Task 

<div align="center">
  <table style="border-collapse: collapse; border: none;">
    <tr>
      <td align="center" style="border: none; padding-right: 20px;">
        <b>DDIM = 4</b><br>
        <img src="https://github.com/user-attachments/assets/7c4d787b-2278-4683-94d5-97a3f6e8c556" width="300"><br>
      </td>
      <td align="center" style="border: none; padding-left: 20px;">
        <b>DDIM = 8</b><br>
        <img src="https://github.com/user-attachments/assets/145d3a87-c982-4f54-afbb-aa63ddc660be" width="300"><br>
      </td>
    </tr>
  </table>
</div>

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.


# Discussion

## Guidance Strength λ<sub>t</sub>
<div align="center">
  <table style="border-collapse: collapse; border: none;">
    <tr>
      <td align="center" style="border: none; padding-right: 20px;">
        <b>Fixed</b> Guidance Ratio λ (VGD)<br>
        <img src="https://github.com/user-attachments/assets/3fb9606f-5dba-48c3-81b4-30b28b5518ee" width="300"><br>
        <sub>Figure #. Lorem Ipsum</sub>
      </td>
      <td align="center" style="border: none; padding-left: 20px;">
        <b>Dynamic</b> Guidance Ratio λ (Ours)<br>
        <img src="https://github.com/user-attachments/assets/9ad0ff10-35e7-40d9-b0e2-8be634aaadf5" width="300"><br>
        <sub>Figure #. Lorem Ipsum</sub>
      </td>
    </tr>
  </table>
</div>

## Correlation between TD error and λ<sub>t</sub>
<div align="center">
  <table style="border-collapse: collapse; border: none;">
    <tr>
      <td align="center" style="border: none; padding-right: 20px;">
        <b>TD Error</b><br>
        <img src="https://github.com/user-attachments/assets/7725dbf1-976d-4c97-a467-26e07b6886b0" width="300"><br>
        <sub>Figure #. TD Error Visualization of Dynamic VGD on the Robomimic Can Task.</sub>
      </td>
      <td align="center" style="border: none; padding-left: 20px;">
        <b>Guidance Strength λ<sub>t</sub></b><br>
        <img src="https://github.com/user-attachments/assets/9ad0ff10-35e7-40d9-b0e2-8be634aaadf5" width="300"><br>
        <sub>Figure #. Guidance Value Visualization of Dynamic VGD on the Robomimic Can Task.</sub>
      </td>
    </tr>
  </table>
</div>
Initially, the critic training is unstable due to high TD error. During the early training phase (up to 500k environment steps), the guidance strength λ<sub>t</sub> gradually decreases, which coincides with the period of high critic instability. As training progresses and the TD error stabilizes, λ<sub>t</sub> increases correspondingly. These results indicate that our dynamic guidance algorithm adapts to the critic's training stability over time.  

## Robustness to DDIM steps 
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.

# Conclusion

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.

# Future Works 
- Distributional RL to capture value uncertainty
- Fine-tuning VLA with Dynamic VGD
  - Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
- Real-world Experiment
  - Improving sample-efficiency
  - Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.

# Reference 
- VGD
- DSRL
- Distributional RL

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
