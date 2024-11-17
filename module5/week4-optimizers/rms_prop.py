import numpy as np

def gradient(W):
    w1, w2 = W
    dw1 = 0.2 * w1
    dw2 = 4 * w2
    return np.array([dw1, dw2], dtype=np.float32)

def rms_prop(w, lr, epochs, beta=0.9, epsilon=1e-8):
    w = np.array(w, dtype=np.float32)
    cache = np.zeros_like(w)
    results = [w.copy()]
    for epoch in range(epochs):
        grad = gradient(w)
        cache = beta * cache + (1 - beta) * grad**2
        w -= lr * grad / (np.sqrt(cache) + epsilon)
        results.append(w.copy())
    return results

if __name__ == '__main__':
    intial_w = [-5, -2]
    lr = 0.3
    epochs = 30
    beta = 0.9
    epsilon = 1e-6
    results = rms_prop(intial_w, lr, epochs)
    for epoch, W in enumerate(results):
        print(f'Epoch {epoch}: {W}')