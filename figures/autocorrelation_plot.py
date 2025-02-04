import numpy as np
import matplotlib.animation as manimation
import matplotlib.pyplot as plt


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1./(1.+ np.exp(-x))

def sigmoid_prime(x: np.ndarray) -> np.ndarray:
    return sigmoid(x)*(1. - sigmoid(x))


def auto_corr(x: np.ndarray) -> np.ndarray:
    ac = []
    for i in range(1,len(x)):
        acv = sum(x[:i]*x[-i:])
        ac.append(acv)
    ac.append(sum(x*x))
    for i in range(1, len(x)):
        acv = sum(x[i:]*x[:-i])
        ac.append(acv)
    return np.stack(ac)


def plot_corr(x: np.ndarray) -> np.ndarray:

    ffmpeg_writer = manimation.writers["ffmpeg"]
    metadata = dict(
        title="autocorr-movie",
        artist="Matplotlib",
        comment="Autokorr.",
    )
    writer = ffmpeg_writer(fps=5, metadata=metadata)
    fig = plt.figure()

    with writer.saving(fig, f"autocorr-movie.gif", 100):
        ac = []
        for i in range(1,len(x)):
            acv = sum(x[:i]*x[-i:])
            ac.append(acv)
            plt.plot(np.concatenate([np.stack([0.]*(len(x))), x, np.stack([0.]*(len(x)))]), 'tab:blue', label='$x_t$')
            plt.plot(np.concatenate([np.stack([0.]*i), x, np.stack([0.]*(len(x)*2-i))]), 'tab:orange', label='$x_{t+k}$')
            plt.plot(np.concatenate([np.stack([0.]*(len(x)//2)), np.stack(ac)]), 'tab:green', label='$c_k$')
            plt.xlabel("$x$")
            plt.legend()
            writer.grab_frame()
            plt.clf()

        ac.append(sum(x*x))
        plt.plot(np.concatenate([np.stack([0.]*(len(x))), x, np.stack([0.]*(len(x)))]), 'tab:blue', label='$x_t$')
        plt.plot(np.concatenate([np.stack([0.]*(len(x))), x, np.stack([0.]*(len(x)))]), 'tab:orange', label='$x_{t+k}$')
        plt.plot(np.concatenate([np.stack([0.]*(len(x)//2)), np.stack(ac)]), 'tab:green', label='$c_k$')
        plt.xlabel("$x$")
        plt.legend()
        writer.grab_frame()
        plt.clf()

        for i in range(1, len(x)):
            acv = sum(x[i:]*x[:-i])
            ac.append(acv)
            plt.plot(np.concatenate([np.stack([0.]*(len(x))), x, np.stack([0.]*(len(x)))]), 'tab:blue', label='$x_t$')
            plt.plot(np.concatenate([np.stack([0.]*(len(x)+i)), x, np.stack([0.]*(len(x)-i))]), 'tab:orange', label='$x_{t+k}$')
            plt.plot(np.concatenate([np.stack([0.]*(len(x)//2)), np.stack(ac)]), 'tab:green', label='$c_k$')
            plt.xlabel("$x$")
            plt.legend()
            writer.grab_frame()
            plt.clf()
    return np.stack(ac)    


if __name__ == '__main__':
    x = np.linspace(-10., 10., 64)
    sp = sigmoid_prime(x)


    # plt.plot(x, sp)
    # plt.show()

    ac = auto_corr(sp)
    # plt.plot(ac)
    # plt.show()
    assert np.allclose(ac, np.correlate(sp,sp, mode='full'))


    plot_corr(sp)

    pass