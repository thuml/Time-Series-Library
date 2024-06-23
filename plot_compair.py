import pandas as pd
import matplotlib.pyplot as plt

def plot_pred():
    train_data = pd.read_csv('train_data.csv')
    test_data = pd.read_csv('test_data.csv')

    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(train_data['x_train'].values, train_data['y_train'].values, 'go', label='f(x)')
    plt.plot(test_data['x_test'].values, test_data['y_pred_kan_8'].values, 'k-', label='KAN(8)')
    plt.plot(test_data['x_test'].values, test_data['y_pred_kan_16'].values, 'r-',label='KAN(16)')
    plt.plot(test_data['x_test'].values, test_data['y_pred_kan_24'].values, 'm-',label='KAN(24)') 

    plt.plot(test_data['x_test'].values, test_data['y_pred_mlp_64'].values, 'b-', label='MLP(64)') 
    plt.plot(test_data['x_test'].values, test_data['y_pred_mlp_128'].values, 'b--', label='MLP(128)') 
    plt.plot(test_data['x_test'].values, test_data['y_pred_mlp_256'].values, 'y-.', label='MLP(256)') 

    plt.title('Xấp xỉ hàm f(x) theo KAN và MLP')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.show()

def plot_loss():
    losses = pd.read_csv('loss_data.csv')
    # Plot the convergence speed
    epochs = 20001
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(range(0, epochs, 100), losses['kan_8_loss'], 'k-', label='KAN(8)')
    plt.plot(range(0, epochs, 100), losses['kan_16_loss'], 'r-', label='KAN(16)')
    plt.plot(range(0, epochs, 100), losses['kan_24_loss'], 'm-', label='KAN(24)')
    plt.plot(range(0, epochs, 100), losses['mlp_64_loss'], 'b-', label='MLP(64)')
    plt.plot(range(0, epochs, 100), losses['mlp_128_loss'], 'b--', label='MLP(128)')
    plt.plot(range(0, epochs, 100), losses['mlp_256_loss'], 'y-.', label='MLP(256)')
    
    plt.title('So sánh tốc độ hội tụ giữa KAN và MLP')
    plt.xlim(-10, 20001)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

plot_pred()
plot_loss()
