import matplotlib.pyplot as plt
import wandb

def plot_and_log(figure_name, x_data, y_data, x_label, y_label, title, legend):
    plt.plot(x_data, y_data, 'y', label=legend)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()
    wandb.log({figure_name: plt})
    plt.close()
