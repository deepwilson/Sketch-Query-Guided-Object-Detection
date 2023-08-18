import matplotlib.pyplot as plt
import json

# Path to your log file
log_file_path = "./checkpoint/log.txt"

# Lists to store epoch and loss values
epochs = []
train_losses = []
test_losses = []

# Read and parse the log file
with open(log_file_path, 'r') as log_file:
    lines = log_file.readlines()
    for line in lines[:150]:
        if line.strip():  # Skip empty lines
            log_entry = json.loads(line)
            epoch = log_entry["epoch"]
            train_loss = log_entry["train_loss"]
            test_loss = log_entry["test_loss"]
            
            epochs.append(epoch)
            train_losses.append(train_loss)
            test_losses.append(test_loss)

# Create the loss vs epochs plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Test Loss vs Epochs')
plt.legend()
plt.grid()

# Save the plot as an image file
plt.savefig('loss_vs_epochs.png')

# Display the plot
# plt.show()
