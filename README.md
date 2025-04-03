# Mitosis
Here we provide the codes for the manuscript "Motorized Chromosome Models of Mitosis" (arXiv:2501.09873).

To simulate the structure of motorized chromosomes, please download "Motorized_mitosis.zip". The initial structures for the simulation refer to "input_interphase.txt" and "mitotic_chromosome.txt", which correspond to the steady states without motor driving and with two types of motors (condensin I & II), respectively. Please create a folder named "Input" in the program folder and place the above initial structure files inside it. At the same time, generate a folder named "Output", where all output files can be accessed after running the program. Our program allows for simulations using random initial configurations, which can be simply selected through the "True" or "False" command in the program. Unless otherwise specified, all steady-state structures are independent of the selection of the above initial structures. 

The representative simulated trajectories for both the puly sequence-distance-dependent model and the spatial-distance-dependent model are attached here in "Output_sequence dependent.zip" and "Output_r dependent.zip", correspondingly. Each compressed file contains at least 18 simulatied trajectories starting from different random initial configurations to demonstrate the repeatability of the simulation. Drag the corresponding.dump file and.topo into the OVITO at the same time to watch the evolution of the simulation trajectory.

In addition, other programs used in the manuscript that process data are also available for reference and use.
