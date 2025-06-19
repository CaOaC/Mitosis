# MITOSIS

Simulation tools for modeling the motorized chromosome.



## ✨ Features



- Simulates the formation process of motor-driven chromosome structure
- Supports both purely sequence-distance-dependent and domain-specific spatial-distance-dependent motor grappling probabilities
- Supports constant-force stretching of the chromosome
- Implements Condensin I / II motor activity models with residence
- Compatible with OVITO for visualization of `.dump` and `.data` trajectory files

## 📦 Dependencies



- [CUDA](https://developer.nvidia.com/cuda-downloads) ≥ 11.2
- [Make](https://www.gnu.org/software/make/) ≥ 4.2.1
- OS: Recommended GNU/Linux (e.g., Ubuntu 20.04 or higher)



## 🚀 Usage



Clone the repository and compile the code:

```bash
git clone https://github.com/CaOaC/Mitosis.git

cd Mitosis/src

# Prepare the input and output folders (准备输入和输出文件夹)
mkdir Input Output

# Compile and run the program (构建并运行程序)
make -j8

./kickModel
```





## 📺 Demo



You can load the following files from the **`Output`** **Output** **Output** **Output** directory into `OVITO` to visualize the simulation trajectory.

- `.dump` file: trajectory animation
- `.data` file: topological structure


<h3>🎞️ movie </h3>





<p>Visualization of chromosome folding process:</p>





<div align="center">





  <img src="./media/demo.gif" width="500"/>





</div>






## 🧾 Information



Here we provide the codes for the manuscript "Motorized Chromosome Models of Mitotic Chromosome Folding" (arXiv:2501.09873).



The initial structures for the simulation refer to "input_interphase.txt" and "mitotic_chromosome.txt", which correspond to the steady states without motor driving and with two types of motors (condensin I & II), respectively. Please create a folder named "Input" in the program folder and place the above initial structure files inside it. At the same time, generate a folder named "Output", where all output files can be accessed after running the program. Our program allows for simulations using random initial configurations, which can be simply selected through the "True" or "False" command in the program. Unless otherwise specified, all steady-state structures are independent of the selection of the above initial structures.



The representative simulated trajectories for both the puly sequence-distance-dependent model and the domain-specific spatial-distance-dependent model are attached at [https://drive.google.com/drive/folders/1t2pTvgPTHxSTNimGVahHoQ1uWVwCQbTx?usp=sharing](https://drive.google.com/drive/folders/1t2pTvgPTHxSTNimGVahHoQ1uWVwCQbTx?usp=sharing) ("Output_sequence dependent.zip" and "Output_r dependent.zip"). Each compressed file contains at least 13 simulated trajectories to ensure repeatability by starting with different random initial configurations. Drag the corresponding ".dump" file and ".data" files into OVITO simultaneously to view the evolution of the simulation trajectory. The kicking numbers for each bead per unit time are recorded in the ".txt" file.



In addition, other programs used in the manuscript that process data are also available for reference and use.