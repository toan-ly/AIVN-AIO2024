import numpy as np

def gradient(W):
    w1, w2 = W
    dw1 = 0.2 * w1
    dw2 = 4 * w2
    return np.array([dw1, dw2], dtype=np.float32)
    
def gd_momentum(w, lr, epochs, beta=0.9):
    w = np.array(w, dtype=np.float32)
    v = np.zeros_like(w)
    results = [w.copy()]
    for epoch in range(epochs):
        grad = gradient(w)
        v = beta * v + (1 - beta) * grad
        w -= lr * v
        results.append(w.copy())
    return results

if __name__ == '__main__':
    intial_w = [-5, -2]
    lr = 0.6
    epochs = 30
    beta = 0.5
    results = gd_momentum(intial_w, lr, epochs, beta)
    for epoch, W in enumerate(results):
        print(f'Epoch {epoch}: {W}')
