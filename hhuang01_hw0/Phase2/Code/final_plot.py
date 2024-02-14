import pandas as pd

data = {
    "Model": ["CNN", "Improved CNN", "ResNet18", "ResNeXt50", "DenseNet"],
    "No of Parameters": [12, 12, 122, 314, 727],
    "Train Accuracy": [54.998, 88.656, 89.556, 70.67, 83.47],
    "Test Accuracy": [46.56, 83.62, 86.51, 68.34, 79.59],
    "Run-time": [0.0149, 0.0104, 0.0103, 0.0079, 0.0159]
}

df = pd.DataFrame(data)

df.set_index('Model', inplace=True)

print(df)