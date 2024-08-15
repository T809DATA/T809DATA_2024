import matplotlib.pyplot as plt
import numpy as np

def plot_multiple():
    '''
    Draws three plots on the same figure by calling
    the function power_plot three times.
    '''
    def power_plot(x_range: np.ndarray, power: int):
        '''
        Plot the function f(x) = x^<power> on the given
        range
        '''
        plt.plot(x_range, np.power(x_range, power), label=f'Power {power}')

    plt.clf() # clear the current figure if there is one

    x_range = np.linspace(-5, 5, 100)
    power_plot(x_range, 1)
    power_plot(x_range, 2)
    power_plot(x_range, 3)

    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    plot_multiple()