
# Publication

If you use this in some academic work, please cite the related publication:

Pire, Taihú; Gonzalez, Hernán; Santano, Emiliano; Terissi, Lucas; Civera, Javier, **An experimental evaluation of feature detectors and descriptors for visual SLAM**. International Journal of Computational Vision and Robotics, 2023. DOI: https://doi.org/10.1504/IJCVR.2023.130648

```bibtex
@article{pire2023experimental,
 author = {Pire, Taih\'{u} and Gonzalez, Hern\'{a}n and Santano, Emiliano and Terissi, Lucas and Civera, Javier},
 title = {An experimental evaluation of feature detectors and descriptors for visual SLAM},
 journal = {International Journal of Computational Vision and Robotics},
 volume = {13},
 number = {3},
 pages = {316--341},
 year = {2023},
 doi = {https://doi.org/10.1504/IJCVR.2023.130648},
 URL = {https://www.inderscienceonline.com/doi/abs/10.1504/IJCVR.2023.130648},
 eprint = {https://www.inderscienceonline.com/doi/pdf/10.1504/IJCVR.2023.130648}
}
```

# Disclaimer
This site and the code provided here are under active development. Even though we try to only release working high quality code, this version might still contain some issues. Please use it with caution.

# Compilation
Docker container is provided to facilitate the compilation and execution

```
docker build . --tag evaluation:latest
```

# Run

Run the docker image
```
docker run --name evaluation --rm -it evaluation:latest
```

OR Run Docker with GUI support:
```
sh run_docker_gui.sh
```

Executable will appear in `<project-path>/build/mainproject`

In there run:

`./Features DetectorName DescriptorName MatcherName [anms] [show]`

Passing the option **anms** enables the Adaptative Non-Maximal Suppression.

Passing the option **show** displays the images with the matches.

## Supported Detectors
- FAST
- ORB
- BRISK
- AGAST
- GFTT
- BAFT
- LOCKY(S)

## Supported Descriptors
- BRIEF
- BRISK
- FREAK
- ORB
- LATCH
- LATCHK\*
- LDB
- BAFT
- BOLD

## Supported Matchers ###
- BFM
- FLANN
- GMS

## Observations ###

**LATCHK** requires **-mavx2** and **-mfma** flags to work, they're currently disabled and
LATCHK's code commented as one of our laptops didn't support **avx2** nor **fma**. If
your hardware supports them, you can uncomment the lines in the options.cmake file an uncomment
that part of the code. (**BAFT** and **LOCKY** also compiled with these optimizations
and didn't work properly, code is not commented though as they work just fine
without this. Again, if **avx2** and **fma** *are* supported, add them to the Makefile).
