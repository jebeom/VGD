# 🚀 Getting Started

### 1. Prerequisites (MuJoCo Setup)
This project requires **MuJoCo 2.1.0**. If you don't have it installed in the default location (`~/.mujoco/mujoco210`), follow these steps:
* **Download and Extract:**
    ```bash
    mkdir -p ~/.mujoco
    wget https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
    tar -xzf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco
    ```
* **Mujoco Dependency:**
  ```bash
  sudo apt update
  sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 libglew-dev \
                     mesa-utils xorg-dev libxrender1 libxext6 libxtst6 \
                     libxi6 libgl1-mesa-dev
  ```
* **Configure Environment Variables:** Append these lines to your `~/.bashrc`:
    ```bash
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$USER/.mujoco/mujoco210/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
    ```
    Then run `source ~/.bashrc`.

---

### 2. Installation & Environment Setup

Follow these steps to clone the repository and install the custom forks of DPPO and Stable Baselines3.

#### Step A: Clone Repository with Submodules
Since this project uses specific versions of DPPO and SB3, you must clone recursively:
```
git clone --recurse-submodules https://github.com/jebeom/Dynamic_VGD.git
cd Dynamic_VGD
```
#### Step B: Create Conda Environment
```
conda create -n dyna_vgd python=3.9 -y
conda activate dyna_vgd
```
#### Step C: Install Custom Submodules(DPPO) (Editable Mode)
```
cd dppo
pip install -e .
pip install -e .[robomimic]
pip install -e .[gym]
cd ..
```
#### Step D: Install Custom Submodules(Stable Baselines3) (Editable Mode)
```
cd stable-baselines3
pip install -e .
cd ..
```

### 📥 Note: Pre-trained Checkpoints
The diffusion policy checkpoints for the Robomimic and Gym experiments can be found [here](https://drive.google.com/drive/folders/1kzC49RRFOE7aTnJh_7OvJ1K5XaDmtuh1?usp=share_link). Download the contents of this folder and place in `./dppo/log`.

---

## Running Dynamic VGD

To track and log your experiments, you need to configure your WandB API key.

1. **Get your API key:** Log in to [wandb.ai/authorize](https://wandb.ai/authorize) to find your key.
2. **Export the API key:** Run the following command in your terminal (replace with your actual key):
   ```bash
   export WANDB_API_KEY=your_actual_api_key_here
   ```
   
To run Dynamic VGD on Robomimic, call
```
python train_vgd.py --config-path=cfg/robomimic/dyna_vgd_square --config-name=dyna_vgd_square_ddim8.yaml
```
💡 **Note**: You can select the appropriate config file based on your experiment needs. For instance, if you want to use 4 DDIM steps or switch the task to 'can', locate the corresponding .yaml file in the cfg/ directory and update the command accordingly.

# 📌 Project Overview

We propose a novel reinforcement learning (RL) based fine-tuning framework designed to significantly enhance the performance of diffusion policies in robotic manipulation. Our approach introduces a **Dynamic Value-Guided Denoising** mechanism that adaptively adjusts the guidance ratio during the denoising process. Unlike static methods, our framework dynamically calibrates the influence of the value function based on critic uncertainty. This results in superior robustness, sample efficiency, and adaptability across diverse environments and complex tasks, overcoming the inherent limitations of standard imitation learning and fixed-guidance strategies.

---

# 🚩 Problem Statement

While diffusion-based policies have demonstrated impressive capabilities in robotic manipulation via **Imitation Learning (IL)**, they face fundamental performance ceilings imposed by the static nature of dataset imitation. To transcend these limits, recent works have introduced RL-based fine-tuning methods:

* **Diffusion Steering with Reinforcement Learning (DSRL):** Enhances performance by optimizing the *initial noise* of the diffusion process using an RL policy.
* **Value-Guided Denoising (VGD):** Attempts more direct steering by leveraging a learned value function (Q-function) to guide the *denoising steps*.

**The Gap:**
Although VGD provides a more granular control mechanism, it relies on a **fixed guidance ratio**. This lack of flexibility restricts the policy's ability to adapt to varying degrees of uncertainty within different states or tasks, often leading to suboptimal convergence or instability in diverse environments.

---

# 🛠️ Method

To address the limitations of fixed guidance, we propose a framework that **dynamically regulates the guidance ratio ($\lambda$)**. We introduce a dynamic scalar, $w_{uncertainty}$, which modulates the base guidance strength based on the critic's reliability.

### 1. Uncertainty Estimation Components
Our dynamic weight is decomposed into two distinct factors: **Local Uncertainty** and **Global Stability**.

- **Local Uncertainty ($w_{local}$):** Measures the critic's confidence regarding the current state-action pair using the variance of an ensemble of 4 critics.

$$
w_{local}(s,a) = \exp\left(-\beta \cdot \frac{\sigma_Q(s,a)}{C_{local}}\right)
$$

  *(High variance -> Low confidence -> Reduced weight)*

- **Global Stability ($w_{global}$):** Reflects the overall stability of the critic during training using the average Temporal Difference (TD) error.

$$
w_{global} = \exp\left(-\beta \cdot \frac{\bar{\delta}_{TD}}{C_{global}}\right)
$$
  
  *(High TD error -> Unstable critic -> Reduced weight)*

### 2. Dynamic Guidance Formulation
We synthesize these components to compute the comprehensive uncertainty weight, which then scales the base guidance ratio ($\lambda_{base}$).

$$
w_{uncertainty} = w_{local}(s,a) \cdot w_{global}
$$

$$
\lambda_{dynamic} = \lambda_{base} \cdot w_{uncertainty}
$$

### 3. Final Denoising Guidance
Finally, this dynamic guidance ratio is applied to the diffusion denoising step. The modified score function (or gradient update) for the action $a_t$ at timestep $t$ is formulated as:

$$
\nabla_{a_t} \log \tilde{p}(a_t|s) \approx \nabla_{a_t} \log p_\theta(a_t|s) + \lambda_{dynamic} \cdot \nabla_{a_t} Q(s, a_t)
$$

By dynamically attenuating the guidance when the critic is uncertain or unstable, our method ensures safe and robust policy improvement, outperforming static VGD approaches.

---

# 🧪 Experimental Results 

Our training pipeline involves (1) pretraining a Diffusion Policy on the designated task and (2) subsequently fine-tuning the pretrained model with reinforcement learning using our method. We compare its performance against two baseline methods: DSRL and VGD.

We evaluate our method on two Robomimic benchmark tasks, Can and Square (see video below).

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


### Can Task 

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

### Square Task 

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

The results show that our method consistently outperforms the baseline approaches. In particular, Dynamic VGD (ours) exhibits faster convergence, resulting in improved sample efficiency. This performance trend remains robust across different DDIM denoising steps, including DDIM = 4 and DDIM = 8.

# 🗨️ Discussion

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

Vanilla VGD uses a fixed guidance ratio throughout training, whereas our method dynamically adjusts the guidance strength over time.


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
Initially, critic training is unstable due to high TD error. During the early training phase (up to 500k environment steps), the guidance strength λ<sub>t</sub> gradually decreases, coinciding with this period of high critic instability. As training progresses and the TD error stabilizes, λ<sub>t</sub> increases accordingly. These results indicate that our dynamic guidance algorithm adapts to the critic's training stability over time.  

# Future Works 
- Distributional RL to capture value uncertainty
  - We can adopt distributional reinforcement learning to explicitly model value uncertainty rather than relying on a deterministic value estimate. This uncertainty-aware value representation provides more informative guidance during policy optimization and is particularly important when fine-tuning diffusion-based policies, where inaccurate value estimates can destabilize training.
- Fine-tuning VLA with Dynamic VGD
  - We can fine-tune a pretrained VLA model using Dynamic VGD. By leveraging distributional value estimates, Dynamic VGD balances exploration and exploitation throughout training, enabling stable optimization even during early phases when critic uncertainty is high.
- Real-world Experiment
  - Our method demonstrates improved sample efficiency compared to baseline approaches, achieving stronger performance with fewer environment interactions.

## Acknowledgements
This repository builds on [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) and [DPPO](https://github.com/irom-princeton/dppo). Parts of code adapted from [DSRL](https://github.com/ajwagen/dsrl), [VGD](https://github.com/DozenDucc/VGD)

--- 

# Contribution of each team member

- **Jebeom Chae** (2025324138) - Jebeom contributed to the project by proposing and implementing Dynamic VGD, running experiments on the Robomimic Benchmark, and creating the presentation materials.
- **Hyunjin Park** (2025314232) - Hyunjin contributed to the project by organizing the team, proposing the initial project idea, and conducting the experiment in robomimic environment.
- **Meedeum Cho** (2025
  

