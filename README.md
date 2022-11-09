# N-Body Simulation using Compute Shaders and the Mesh Method

This is a small project I did with the goal of simulation a lot of bodies and their gravitational interaction on my personal machine. The approach is based on the mesh method and the gravitational field is solved using a FFT implemented in compute shaders. The gradient information of the forces is also used for more accurate stepping.
Overall, for 10 million parameters and a mesh size of 2048x2048 this runs with about 8fps on a GTX 1070ti. Unfortunately, I use atomic float operations, which are only available for Nvidia GPUs.
The shader pipeline is called using pythen with the modernGL package.

Videos of the simulation can be found here:

https://youtube.com/shorts/ZYrsbhrFhmA

https://youtube.com/shorts/XWciKnHgzkg

https://youtube.com/shorts/7eAeNYhOIso
