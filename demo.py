import numpy as np
import os
import argparse
# import asp_gen.run_occam as run_occam 
from asp_gen.run_occam import get_prediction, get_attr, slope_with_map

pcl = np.load('demo_pcl.npy')
pert_boxes, pert_labels, pert_scores = get_prediction(pcl)
map, pert_boxes = get_attr(pcl)
corrupt_pcl = slope_with_map(pcl, map, 0.002, pert_boxes)

corrupt_boxes, corrupt_labels, corrupt_scores = get_prediction(corrupt_pcl)
print("the label of clean data: ", pert_labels)
print("the label of corrupt data: ", corrupt_labels)

np.save('corrupt_pcl.npy', corrupt_pcl)




