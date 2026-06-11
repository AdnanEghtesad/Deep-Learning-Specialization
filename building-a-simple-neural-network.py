import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time

CPU_Parallelism = 1
GPU_Parallelism = 0 
assert (CPU_Parallelism + GPU_Parallelism == 1), "You can use either CPU or GPU, but not both or neither."

if(CPU_Parallelism):
    device = torch.device("cpu")
    # Use 8 CPU cores
    torch.set_num_threads(8)
    # Optional: control inter-op parallelism
    torch.set_num_interop_threads(8)
elif(GPU_Parallelism):
    device = torch.device("cuda")
    print("Torch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())
    print(f"Using device: {device}")

tic = time.time()

   

#import helper_utils

# This line ensures that your results are reproducible and consistent every time.
torch.manual_seed(42)
# Distances in miles for recent bike deliveries
distances = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
# Corresponding delivery times in minutes
times = torch.tensor([[6.96], [12.11], [16.77], [22.21]], dtype=torch.float32)
# Create a model with one input (distance) and one output (time)
model = nn.Sequential(nn.Linear(1, 1))
# Loss function
loss_function = nn.MSELoss()
# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)
# Training loop
loss_history = []
for epoch in range(500):
    # Reset the optimizer's gradients
    optimizer.zero_grad()
    # Make predictions (forward pass)
    outputs = model(distances)
    # Calculate the loss
    loss = loss_function(outputs, times)
    # Calculate adjustments (backward pass)
    loss.backward()
     # Save loss
    loss_history.append(loss.item())
    # Update the model's parameters
    optimizer.step()
    # Print loss every 50 epochs
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}: Loss = {loss.item()}")


toc = time.time()
print(f"Elapsed time = {toc - tic:.6f} seconds")

# Get model predictions
with torch.no_grad():
    predicted_times = model(distances)        
# Plot
# Convert tensors to NumPy arrays
x = distances.numpy().flatten()
y_actual = times.numpy().flatten()
y_pred = predicted_times.numpy().flatten()
plt.figure(figsize=(8, 5))
plt.scatter(x, y_actual, label='Actual Data')
plt.plot(x, y_pred, label='Model Prediction')
plt.xlabel('Distance (miles)')
plt.ylabel('Delivery Time (minutes)')
plt.title('Delivery Time vs Distance')
plt.grid(True)
plt.legend()   
plt.show()     

plt.figure(figsize=(8,5))
plt.plot(range(1, len(loss_history)+1), loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training Loss vs Epoch')
plt.grid(True)
plt.show()