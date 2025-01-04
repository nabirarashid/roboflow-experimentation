from roboflow import Roboflow
import os
import json
import time

# Create a Roboflow object
# Access an environment variable that stores the API key
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])

# Access the workspace
workspace = rf.workspace("nabirarashid")

# Create a list of projects to apply the configurations
projects = ["rock-paper-scissors-c5y1x"]

# Using starter configuration to kick off a training job
# Has info on augmentation & preprocessing steps

# waiting mechanism for status of the project
def wait_for_dataset_ready(project_item, polling_interval=10):
    while True:
        status = project_item.status()
        if status.get("progress") == 100:
            print("Dataset version is ready!")
            break
        else:
            print(f"Dataset is still generating. Progress: {status.get('progress')}%")
            time.sleep(polling_interval)

# Import project, generate a version & training the version of project
def generate_train(project,configuration):
    rf_project = workspace.project(project)
    version_number = rf_project.generate_version(configuration)
    project_item = workspace.project(project).version(version_number)
    wait_for_dataset_ready(project_item)
    project_item.train()

with open("configurations/starter.json","r") as f:
    configuration = json.load(f)

generate_train(projects[0],configuration)