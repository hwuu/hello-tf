#
# This Python script does the following things:
#     1. Create a new workspace with a specified name, if it does not exist, connect to it and write its config to a local file.
#     2. Create a compute with a specified name in the workspace for training jobs to run on.
#     3. Register an existing datastore to the workspace.
#

import os
import sys
import azureml.core
from azureml.core import Workspace
from azureml.core import Datastore
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

#
# Prepare the workspace.
#

subscription_id = os.getenv("SUBSCRIPTION_ID", default="f36a6329-7382-4b2e-b386-452fecdfcd73")
resource_group = os.getenv("RESOURCE_GROUP", default="aml-test")
workspace_name = os.getenv("WORKSPACE_NAME", default="aml-test-ws")
workspace_region = os.getenv("WORKSPACE_REGION", default="eastus2")

ws = None
try:
    print("Connecting to workspace '%s'..." % workspace_name)
    ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
except:
    print("Workspace not accessible. Creating a new one...")
    try:
        ws = Workspace.create(
            name = workspace_name,
            subscription_id = subscription_id,
            resource_group = resource_group, 
            location = workspace_region,
            create_resource_group = True,
            exist_ok = True)
    except:
        print("Failed to connect to workspace. Quit with error.")
        sys.exit(1)
print(ws.get_details())

ws.write_config()

#
# Prepare the compute in the workspace.
#

cluster_name = "gpucluster"
try:
    ct = ComputeTarget(workspace=ws, name=cluster_name)
    print("Found existing cluster '%s'. Skip." % cluster_name)
except ComputeTargetException:
    print("Creating new cluster '%s'..." % cluster_name)
    compute_config = AmlCompute.provisioning_configuration(vm_size="STANDARD_NC6", min_nodes=0, max_nodes=1)
    ct = ComputeTarget.create(ws, cluster_name, compute_config)
    ct.wait_for_completion(show_output=True)
print(ct.get_status().serialize())

#
# Register an existing datastore to the workspace.
#

datastore_name = "hellotfstore"
if datastore_name not in ws.datastores:
    Datastore.register_azure_blob_container(
        workspace=ws, 
        datastore_name=datastore_name,
        container_name="hello-tf",
        account_name="wuhamltestsa",
        account_key="LBpyUOlJT/wbiHQReiwY1EB3WhDF3Sn2STia4UY//SkMWerh08M0QjhImmQ8TwCrmvDfq0tVtB3xF9mxZFiMXA=="
    )
    print("Datastore '%s' registered." % datastore_name)
else:
    print("Datastore '%s' has already been regsitered." % datastore_name)

# (END)
