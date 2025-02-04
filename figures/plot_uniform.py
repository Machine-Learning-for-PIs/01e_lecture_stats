import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib as tikz

if __name__ == '__main__':
    
    def uniform_pdf(x: float, a: float = 0, b: float = 1):
        if a < x < b:
            return 1./(b-a)
        else:
            return 0. 

    x = np.linspace(-2, 2, 100)
    updf = [uniform_pdf(vx) for vx in x]
    plt.plot(x, updf)
    plt.xlabel('x')
    plt.ylabel('pdf(x)')
    tikz.save('uniform_pdf.tex', standalone=True)
    plt.show() 