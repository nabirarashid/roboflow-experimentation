from roboflow import Roboflow

# Initialize Roboflow with your API key
rf = Roboflow(api_key="ROBOFLOW_API_KEY")

# Access the project and model
project = rf.workspace("nabirarashid").project("rock-paper-scissors-sxsw-rtuhx")
model = project.version(4).model

# Infer on a local image
prediction = model.predict("image.jpg", confidence=40, overlap=30)

# Print the prediction as JSON
print(prediction.json())

# Visualize and save the prediction
prediction.save("prediction2.jpg")

# Optional: Plot the prediction (if running in an interactive environment)
prediction.plot()
