# ASP

This is the official repository of ASP[Attribution-based Scanline Perturbation Attack On 3D Detectors Of LiDAR Point Clouds].

The origin point cloud and perturbated sample is shown in the following figures
<center class="half">
<img src="figs/before_corrupt.png" width=200/><img src="figs/after_corrupt.png" width=200/>
</center>

##  Requirements

* linux(tested with Ubuntu 20.04 LTS)
* Python(tested with Ver.3.8)
* Cuda(tested with Ver.11.3)
* Spconv(tested with spconv-cu113 Ver.2.1.21)

## Instsall required packages
<code>pip install -r requirements.txt </code>

## Set up pcdet
<code>Python setup.py develop </code>

## Run demo
<code>Python demo.py </code>



