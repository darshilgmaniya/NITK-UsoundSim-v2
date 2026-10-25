import numpy as np
#data = np.load("nitk_usoundsim/acoustic_propagation_outputs.npz")
#print(data["received_pressure"])

'''data = np.load("nitk_usoundsim/part1_forward_propagation_outputs.npz")
print(data["description"])     
print(data.files)''' 


data = np.load("nitk_usoundsim/part2_return_propagation_outputs.npz")
print(data["description"])     
print(data["reflected_pressure_input"])
print(data["reflected_pressure_input"].shape)    
print(data["reflected_pressure_input_row_labels"]) 