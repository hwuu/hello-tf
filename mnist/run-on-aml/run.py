#
# This Python script submits the specific job to AML.
#
# We have the following assumptions:
#     1. A workspace has been created in advance.
#     2. An existing compute (either a GPU- or a CPU-cluster) has been created in the workspace.
#     3. An existing external datastore has been registered to the workspace.
#

import os
import sys
import azureml.core
from azureml.core import Workspace
from azureml.core import Datastore
from azureml.core import Experiment
from azureml.core.compute import ComputeTarget
from azureml.core.runconfig import ContainerRegistry
from azureml.train.estimator import Estimator
from azureml.widgets import RunDetails

#
# Get the workspace, compute target, and datastore we prepared previously.
#

ws = Workspace.from_config()
ct = ComputeTarget(workspace=ws, name="cpucluster-II")
ds = Datastore(workspace=ws, name="hellotfstore")

#
# Create an estimator.
#

# Single node
est_1 = Estimator(
    compute_target=compute_target,
    use_gpu=False,
    node_count=1,
    pip_packages=['tensorflow==1.13.1'],
    source_directory="../",
    entry_script="mnist-mlp.py",
    script_params={
        "--data-dir": ds.path("data/mnist").as_mount()
    }
)

# Distributed with PS architecture
#est_2 = ...


# Distributed with Horovod
est_3 = Estimator(
    compute_target=compute_target,
    use_gpu=False,
    node_count=2,
    process_count_per_node=2,
    distributed_backend='mpi',
    pip_packages=['tensorflow==1.13.1', 'horovod'],
    source_directory="../",
    entry_script="mnist-mlp-dist-hvd.py",
    script_params={
        "--data-dir": ds.path("data/mnist").as_mount()
    }
)

#
# Run the job (i.e. estimator) in an experiment.
#

run = Experiment(workspace=ws, name='aml-demo-2').submit(est)
if 'ipykernel' in sys.modules:
    RunDetails(run).show()
else:
    print(run)

# (END)
