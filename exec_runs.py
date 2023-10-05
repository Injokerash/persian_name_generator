import subprocess
import numpy as np


lr_range = 10**np.linspace(-4, -1, 7)
epochs = 60
hidden_size_range = [50, 150, 250]
padding_range = [2, 4, 6]
embedding_dim_range = [2, 4, 5, 6, 8]
batch_size_range = [128, 256, 512]


for learning_rate in lr_range:
    for hidden_size in hidden_size_range:
        for padding in padding_range:
            for embedding_dim in embedding_dim_range:
                for batch_size in batch_size_range:
                    res = subprocess.run(f"mlflow run -e MLP . -P padding={padding} -P hidden_size={hidden_size} -P epochs={epochs} -P embedding_dim={embedding_dim} -P learning_rate={learning_rate} -P batch_size={batch_size}", shell=True)
                    print(res)