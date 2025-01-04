# YOOO THIS WORKS LETS GOOO
from roboflow import Roboflow
import os

# Initialize Roboflow with your API key
api_key = os.environ.get('ROBOFLOW_API_KEY')
rf = Roboflow(api_key=api_key)

# Access the workspace
workspace = rf.workspace("nabirarashid")

# Create a list of projects to apply the configurations
projects = ["rock-paper-scissors-sxsw-rtuhx"]

# Function to generate a version and train the version of the project
def generate_train(project):
    rf_project = workspace.project(project)
    version_number = 4
    project_item = rf_project.version(version_number)
    project_item.train()

# Train the specified project
generate_train(projects[0])