import numpy as np

def gradient(W):
    w1, w2 = W
    dw1 = 0.2 * w1
    dw2 = 4 * w2
    return np.array([dw1, dw2], dtype=np.float32)

def adam(w, lr, epochs, beta1=0.9, beta2=0.999, epsilon=1e-6):
    w = np.array(w, dtype=np.float32)
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    results = [w.copy()]
    for epoch in range(epochs):
        grad = gradient(w)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**(epoch + 1))
        v_hat = v / (1 - beta2**(epoch + 1))
        w -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
        results.append(w.copy())
    return results

if __name__ == '__main__':
    intial_w = [-5, -2]
    lr = 0.2
    epochs = 30
    results = adam(intial_w, lr, epochs)
    for epoch, W in enumerate(results):
        print(f'Epoch {epoch}: {W}')