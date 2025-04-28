import matplotlib.pyplot as plt

def plot_data(data):
    plt.plot(data['date'], data['value'])
    plt.show()

def save_plot(data, filepath):
    plt.plot(data['date'], data['value'])
    plt.savefig(filepath)
