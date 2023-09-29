import torch
import time
from mpi4py.MPI import COMM_WORLD as pycomm
from cupy.cuda import nccl

rank = pycomm.rank
world_size = pycomm.size

torch.cuda.set_device(rank % 8)

ids = pycomm.bcast([nccl.get_unique_id() for _ in range(world_size)] if rank == 0 else None, root=0)

pycomm.Barrier()
start = time.time()
comms = []
for i in range(world_size):
    comm = nccl.NcclCommunicator(world_size, ids[i], rank)
    comms.append(comm)

pycomm.Barrier()
duration = time.time() - start
print(f"CREATED {len(comms)} comms in {duration:0.3f}s", flush=True)



