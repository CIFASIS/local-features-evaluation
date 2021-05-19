## Compilation ##
Docker container is provided to facilitate the compilation and execution

```
docker build . --tag evaluation:latest
```
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

### Supported Detectors: ###
- FAST
- ORB
- BRISK
- AGAST
- GFTT
- BAFT
- LOCKY(S)

### Supported Descriptors: ###
- BRIEF
- BRISK
- FREAK
- ORB
- LATCH
- LATCHK\*
- LDB
- BAFT
- BOLD

### Supported Matchers: ###
- BFM
- FLANN
- GMS

### Observations: ###

**LATCHK** requires **-mavx2** and **-mfma** flags to work, they're currently disabled and
LATCHK's code commented as one of our laptops didn't support **avx2** nor **fma**. If
your hardware supports them, you can uncomment the lines in the options.cmake file an uncomment
that part of the code. (**BAFT** and **LOCKY** also compiled with these optimizations
and didn't work properly, code is not commented though as they work just fine
without this. Again, if **avx2** and **fma** *are* supported, add them to the Makefile).
