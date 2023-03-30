# IR-multichannel
Calculate multichannel IR files through deconvolution of a recorded sweep. With a jupyter notebook example (Python).

You need python and Jupyter (https://jupyter.org/install).

How to use this code:

Download and copy all code to a folder in your computer. 

Run jupyter from a terminal "jupyter notebook" , open the jupyter notebook example and execute the first cell. 

A mono sweep is included among the code. Record the sweep in a room with a microphone (mono, stereo or 4 channels). Run the example code defining the number of channels (see source code).

This code generates IR files with the same number of channels and order than the recorded sweep you are using.

You can use the calculated .wav file with plugins like MatrixConv (https://leomccormack.github.io/sparta-site/docs/plugins/sparta-suite/#matrixconv) in combination with HO-SIRR (https://leomccormack.github.io/sparta-site/docs/plugins/hosirr/) for adding an ambisonics room to your track in real time. 
