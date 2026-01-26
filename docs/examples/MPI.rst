ase_uhal with MPI4Py
=====================

ase_uhal is compatible with mpi4py, to allow multiple biased MD calculations to occur simultaneously. This is achieved in a fully asynchronous manner, which means that communication between ranks is minimised during the MD runs. 

When a structure is selected on one process, it also communicates the selected structure to all other processes which share a communicator (by default, `COMM_WORLD` is used). When `resample_committee()` is called, the calling process first checks for any messages from other ranks, and adds any recieved structures to its internal linear system before continuing with the resampling. In this way, each rank is able to learn from structures selected by each other rank.

The energy, force, and stress weights are also copied into the message whenever a structure is sent. This allows the different ranks to maintain synchronised linear systems (and thus be approximations to the same distribution).

The committee calculator also has the `sync()` function, which places a recieve operation of all messages in between two MPI barriers. In this way, we ensure that all messages have been sent and recieved, and that all ranks are at the same position of the code.


Example
--------
.. literalinclude:: mpi_example.py