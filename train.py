import torch
import torch.optim as optim
import torch.nn.functional as F
from model import SharedFeedForwardNN
import matplotlib.pyplot as plt

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, weight_decay, alfa, beta, save_dir):

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    train_total_losses = []
    val_total_losses = []
    train_reg_losses = []
    train_class_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()

        running_reg_loss = 0.0
        running_class_loss = 0.0
        running_total_loss = 0.0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad() 

            reg_output, logits = model(inputs)

            reg_loss = F.mse_loss(reg_output.squeeze(), labels[:,0])
            class_loss = F.cross_entropy(logits, labels[:,1].long())
            total_loss = alfa*reg_loss + beta*class_loss
            
            total_loss.backward()  
            optimizer.step() 
            
            running_reg_loss += reg_loss.item()
            running_class_loss += class_loss.item()
            running_total_loss += total_loss.item()
        
        train_reg_loss = running_reg_loss / len(train_loader)
        train_class_loss = running_class_loss / len(train_loader)
        train_total_loss = running_total_loss / len(train_loader)

        train_total_losses.append(train_total_loss)
        train_reg_losses.append(train_reg_loss)
        train_class_losses.append(train_class_loss)

        print(f'\nEpoch [{epoch+1}/{num_epochs}], Regression loss: {train_reg_loss:.4f}')
        print(f'Epoch [{epoch+1}/{num_epochs}], Classification Loss: {train_class_loss:.4f}')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_total_loss:.4f}')

        model.eval()

        running_total_loss = 0.0
        total_samples = 0 
        total_correct = 0 
    
        with torch.no_grad(): 
            for inputs, labels in val_loader:
                reg_output, logits = model(inputs)

                labels = labels.to(torch.int64)

                reg_loss = F.mse_loss(reg_output.squeeze(), labels[:,0])
                class_loss = F.cross_entropy(logits, labels[:,1]) 
                total_loss = alfa * reg_loss + beta * class_loss
                
                running_total_loss += total_loss.item()

                probas = F.softmax(logits, dim=1)
                _, predicted_class = torch.max(probas, 1)
                total_samples += labels[:,1].size(0)
                total_correct += (predicted_class == labels[:,1]).sum().item()

        avg_total_loss_val = running_total_loss / len(val_loader)
        accuracy_val = 100 * total_correct / total_samples

        val_total_losses.append(avg_total_loss_val)
        val_accuracies.append(accuracy_val)

        print(f'Validation - Total Loss: {avg_total_loss_val:.4f}, Accuracy: {accuracy_val:.2f}%')

    train_loss_avg = sum(train_total_losses) / num_epochs
    val_loss_avg =  sum(train_total_losses)/ num_epochs
    reg_loss_avg = sum(train_reg_losses) / num_epochs
    class_loss_avg = sum(train_class_losses) / num_epochs

    val_acc_avg = sum(val_accuracies) / num_epochs

    print(f"\nAverage Total Loss (Train): {train_loss_avg:.4f}")
    print(f"Average Regression Loss (Train): {reg_loss_avg:.4f}")
    print(f"Average Classification Loss (Train): {class_loss_avg:.4f}")
    print(f"Average Total Loss (Validation): {val_loss_avg:.4f}")
    print(f"Average Accuracy (Validation): {val_acc_avg:.2f}%")

    # Generate and save the plots
    plot_metrics(train_total_losses, val_total_losses, train_reg_losses, train_class_losses, num_epochs, save_dir)


def plot_metrics(train_total_losses, val_total_losses, train_reg_losses, train_class_losses, num_epochs, save_dir):
    
    max_loss = max(max(train_total_losses), max(val_total_losses))
    max_train_loss = max(max(train_reg_losses), max(train_class_losses))

    # Plot 1: Train Loss vs Validation Loss
    plt.figure(figsize=(8, 6))
    plt.plot(train_total_losses, label='Train Loss', color='blue')
    plt.plot(val_total_losses, label='Validation Loss', color='orange')
    plt.title('Train Loss vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xlim(0, num_epochs-1)
    plt.ylim(0, max_loss*1.1)  
    plt.legend()
    plt.grid()
    plt.savefig(f"{save_dir}/train_vs_validation_loss.png")
    plt.close()

    # Plot 2: Regression Loss vs Classification Loss (in training)
    plt.figure(figsize=(8, 6))
    plt.plot(train_reg_losses, label='Regression Loss (Train)', color='green')
    plt.plot(train_class_losses, label='Classification Loss (Train)', color='red')
    plt.title('Regression Loss vs Classification Loss')
    plt.xlabel('Epochs')
    plt.xlim(0, num_epochs-1)
    plt.ylim(0, max_train_loss*1.1) 
    plt.legend()
    plt.grid()
    plt.savefig(f"{save_dir}/regression_vs_classification_loss.png")
    plt.close()
