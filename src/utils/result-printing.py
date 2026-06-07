import matplotlib.pyplot as plt

epochs = [1,2,3,4,5]
loss_v8s  = [5.297, 4.571, 4.207, 4.194, 3.841]
loss_v11s = [5.020, 4.600, 4.350, 4.100, 3.850]  # replace with your true sums

lr_v8s  = [0.01,   0.01,   0.01,    0.0025, 0.0025]
lr_v11s = [0.01,   0.01,   0.0025,  0.0025, 0.00125]

# Plot Loss
plt.figure(figsize=(8,5))
plt.plot(epochs, loss_v8s,  '-o', label='YOLOv8s')
plt.plot(epochs, loss_v11s, '-o', label='YOLOv11s')
plt.title("Training Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.show()

# Plot LR
plt.figure(figsize=(8,5))
plt.plot(epochs, lr_v8s,  '-o', label='YOLOv8s')
plt.plot(epochs, lr_v11s, '-o', label='YOLOv11s')
plt.title("Learning Rate vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.grid(True)
plt.legend()
plt.show()
