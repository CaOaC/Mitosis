# MITOSIS

Simulation tool for modeling motorized chromosome.

## 简介

Here we provide the codes for the manuscript "Motorized Chromosome Models of Mitosis" (arXiv:2501.09873).



To simulate the structure of motorized chromosomes, please download "Motorized_mitosis.zip". The initial structures for the simulation refer to "input_interphase.txt" and "mitotic_chromosome.txt", which correspond to the steady states without motor driving and with two types of motors (condensin I & II), respectively. Please create a folder named "Input" in the program folder and place the above initial structure files inside it. At the same time, generate a folder named "Output", where all output files can be accessed after running the program. Our program allows for simulations using random initial configurations, which can be simply selected through the "True" or "False" command in the program. Unless otherwise specified, all steady-state structures are independent of the selection of the above initial structures.



The representative simulated trajectories for both the puly sequence-distance-dependent model and the spatial-distance-dependent model are attached at [https://drive.google.com/drive/folders/1t2pTvgPTHxSTNimGVahHoQ1uWVwCQbTx?usp=sharing](https://drive.google.com/drive/folders/1t2pTvgPTHxSTNimGVahHoQ1uWVwCQbTx?usp=sharing) ("Output_sequence dependent.zip" and "Output_r dependent.zip"). Each compressed file contains at least 13 simulated trajectories to ensure repeatability by starting with different random initial configurations. Drag the corresponding ".dump" file and ".data" into the OVITO at the same time to watch the evolution of the simulation trajectory. The kicking numbers for each bead per unit time are recorded in the ".txt" file.



In addition, other programs used in the manuscript that process data are also available for reference and use.

## ✨ Features



- 模拟马达染色质结构的生成过程
- 序列依赖和距离依赖
- 支持恒力拉伸染色质
- 支持 Condensin I / II 动力模型
- 支持使用 OVITO 可视化 `.dump` 和 `.data` 轨迹文件

## 📦 Dependencies



- [CUDA](https://developer.nvidia.com/cuda-downloads) ≥ 11.2
- [GNU Make](https://www.gnu.org/software/make/) ≥ 4.2.1
- 操作系统：建议使用 GNU/Linux（如 Ubuntu 20.04 或更高）



## 🚀 Usage

Clone the repository and compile the code:



```bash
git clone https://github.com/CaOaC/Mitosis.git

# 准备输入和输出文件夹
mkdir Input Output

cd Mitosis

# 构建并运行程序
make -j8

./kicktauleap
```





## 📺 Demo



你可以在 OVITO 中加载以下文件观察模拟轨迹：



- `.dump` 文件：轨迹动画
- `.data` 文件：拓扑结构
- `.txt` 文件：单位时间每个 beads 的抓取次数

![]()

