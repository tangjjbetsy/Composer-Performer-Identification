import matplotlib.pyplot as plt
import numpy as np

records = np.load("records_resnet18_square_final_2.npy", allow_pickle=True)
records = records.tolist()
loss = records['loss']
dev_loss = records["dev_loss"]
print(records["acc"][-1])
print(records["dev_acc"][-1])

plt.plot(np.arange(len(loss)), loss)
plt.plot(np.arange(len(loss)), dev_loss)
plt.legend(["training loss", "validation loss"])
plt.title("Trainig acc: 99.59839357429718 \n Validation acc: 76.92307692307692")

plt.savefig("2.png")