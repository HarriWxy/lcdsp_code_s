# Language-Controlled Diverse Style Policies

Welcome to the official PyTorch implementation of the paper:

"Complex Instruction Following with Diverse Style Policies in Football Games" 

This repository includes the code and model for grf environment inference. The training code is currently being refactored and organized for better readability and reproducibility.

Here is an overview of the inference process:

<img src="figures\Inference_Process.png" alt="LCDSP_inference" width="800">

Here is an overview of the style interpreter:

<img src="figures\style_interpreter.png" alt="LCDSP_style_interpreter" width="800">

## Environment

Check details in Google Research Football [football](https://github.com/google-research/football/tree/9a9e35bcd1929a82c2b91eeed777e5e571f29d38)

## GRF Installation

Full details of the installation can be found in the [Compiling Google Research Football Engine](https://github.com/google-research/football/blob/master/gfootball/doc/compile_engine.md#windows), we provide installation guide for windows platforms here.

### 1. Create conda environment

Create a new conda environment and install the required packages.

```
conda create -n grf python=3.8
conda activate grf
# depend on your cuda versiopn
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```

Other packages can be installed using the following command.

```python
pip install --upgrade pip setuptools wheel
pip install psutil
pip install gym==0.16.0
```

### 2. Install dependency

- [Git](https://git-scm.com/downloads/win)

- [Visual Studio 2019 Community Edition](https://visualstudio.microsoft.com/zh-hans/downloads/) (make sure to select "Desktop development with C++" component)

- [CMake](https://cmake.org/download/)

- [vcpkg](https://github.com/microsoft/vcpkg)

First, install Git and VS2019. Then, follow the steps below to install CMake and vcpkg.

CMake is installed using the MSI installer, with the option to add CMake to the environment variables automatically selected during installation. After installation, verify whether CMake is included in the environment variables.

>cmake --version

If CMake does not appear, but "C:\Program Files\CMake\bin" is already listed in the "Path" variable under "System variables" in the environment variables window, a system restart is required.

If you prefer not to restart, you can use temporary environment variables. Open a new CMD window and execute the command, ensuring not to use Bash or PowerShell, as the path format differs.

```
set PATH=C:\Program Files\CMake\bin;%PATH%
cmake --version
```

Install vcpkg in the same CMD window; otherwise, you'll need to set the temporary environment variables again.

```
cd C:\dev
git clone https://github.com/microsoft/vcpkg.git
.\vcpkg\bootstrap-vcpkg.bat
```

### 3. Install GRF in conda environment

```python
python -m pip install gfootball
```

### 4. Copy custom scenarios to conda environment

To use our custom-configured scenario, it needs to be copied into the scenario folder of the gfootball package within the conda grf environment. Run the script using a bash command, or manually perform the copy operation.

```bash 
cp -r env/random_ball_scenarios/* your_conda_path/envs/grf/Lib/site-packages/gfootball/scenarios
```

## Installation of dependencies related to inference code

```bash
pip install -r requirements.txt
```

### 5. Download the pre-trained language model

download the pre-trained language model from [google_dirve](https://drive.google.com/file/d/1C_sFRU4pcZQiC7o3JVSo3A2mcTNXN8eY/view?usp=drive_link) and place it in the `language_models` folder.

# Run a game

After starting the game, make sure to click elsewhere with the mouse, and avoid clicking on the game screen afterward to prevent it from freezing.

Use Ctrl+C in the command line to close the game and the input subprocess.


The following command will run using a pre-configured style parameter, and in the 5v5 mode, the opponent will also use this style parameter.

The team wearing yellow uniforms is my_ai, the team wearing blue uniforms is the opponent.

The opponent can choose to engage in self-play using lcdsp_two_player or compete against noop_AI and buildin_AI.

```python
python run_log.py --my_ai lcdsp_5v5 --opponent lcdsp_5v5 --env football_5v5_malib --agent_type agents_5v5
```

## Contorl policy with style-parameter

<img src="figures\param_control.gif" width="1000" alt="param control">
<br>

The corresponding style parameters can be adjusted through Box UI input parameters. You only need to set style_input to True in the arguments. An input box will appear shortly thereafter. 

After launching the game environment, enter the desired style parameters and simply click submit, you can change the style parameters while the environment is running. You can also modify the initial style parameters in the corresponding JSON file within the *base_style_parameters* folder before launching.

The team wearing yellow uniforms is the instruction-controlled team, attacking towards the right side of the field. 

The opponent can choose to engage in self-play using lcdsp_5v5 or compete against noop_AI and buildin_AI.

```python
python run_log.py --my_ai lcdsp_5v5 --opponent lcdsp_5v5 --env football_5v5_malib --agent_type agents_5v5 --style_input True
```

- In the 5v5 environment, all parameters can be freely combined, allowing you to configure them according to your preferences to create the desired style.

## Contorl policy with natural-language instructions

<img src="figures\language_control.gif" width="1000" alt="language control">
<br>

Natural language input can be used to direct the multi-style policy to follow instructions.

You can select the instructions used in the paper from language_control/scenario/instructions, or input your own desired instruction.

If a command cannot be parsed into style parameters, it will revert to the default style.

The opponent can choose to engage in self-play using lcdsp_5v5 or compete against buildin_AI.

```python
python run_log.py --my_ai lcdsp_5v5 --opponent lcdsp_5v5 --env football_5v5_malib --agent_type agents_5v5 --language_input True
```

The following diagram illustrates the behaviors of the Counter Attack and Tiki-Taka tactics in 5v5 scenario.

<img src="figures\two_tactics.png" alt="5v5 tactics" width="1000">


# Train your own policy
The training code is currently being refactored and organized for better readability and reproducibility.