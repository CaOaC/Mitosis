# MITOSIS

Simulation tool for modeling motorized chromosome.



## ✨ Features



- Simulates the formation process of motor-driven chromosome structure
- Supports both sequence-dependent and distance-dependent motor grabbing probabilities
- Supports constant-force stretching of chromosome
- Implements Condensin I / II motor activity models
- Compatible with OVITO for visualization of `.dump` and `.data` trajectory files

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



You can load the following files from the **`Output`** directory into `OVITO` to visualize the simulation trajectory.

- `.dump` file: trajectory animation

- `.data`

 file: topological structure  



![](./media/demo.gif)

## 🧾 Information



Here we provide the codes for the manuscript "Motorized Chromosome Models of Mitosis" (arXiv:2501.09873).



The initial structures for the simulation refer to "input_interphase.txt" and "mitotic_chromosome.txt", which correspond to the steady states without motor driving and with two types of motors (condensin I & II), respectively. Please create a folder named "Input" in the program folder and place the above initial structure files inside it. At the same time, generate a folder named "Output", where all output files can be accessed after running the program. Our program allows for simulations using random initial configurations, which can be simply selected through the "True" or "False" command in the program. Unless otherwise specified, all steady-state structures are independent of the selection of the above initial structures.



The representative simulated trajectories for both the puly sequence-distance-dependent model and the spatial-distance-dependent model are attached at [https://drive.google.com/drive/folders/1t2pTvgPTHxSTNimGVahHoQ1uWVwCQbTx?usp=sharing](https://drive.google.com/drive/folders/1t2pTvgPTHxSTNimGVahHoQ1uWVwCQbTx?usp=sharing) ("Output_sequence dependent.zip" and "Output_r dependent.zip"). Each compressed file contains at least 13 simulated trajectories to ensure repeatability by starting with different random initial configurations. Drag the corresponding ".dump" file and ".data" into the OVITO at the same time to watch the evolution of the simulation trajectory. The kicking numbers for each bead per unit time are recorded in the ".txt" file.



In addition, other programs used in the manuscript that process data are also available for reference and use.