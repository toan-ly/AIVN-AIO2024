import numpy as np

def gradient(W):
    w1, w2 = W
    dw1 = 0.2 * w1
    dw2 = 4 * w2
    return np.array([dw1, dw2], dtype=np.float32)
    
def gd(w, lr, epochs):
    w = np.array(w, dtype=np.float32)
    results = [w.copy()]
    for epoch in range(epochs):
        grad = gradient(w)
        w -= lr * grad
        results.append(w.copy())
    return results

if __name__ == '__main__':
    intial_w = [-5, -2]
    lr = 0.4
    epochs = 30
    results = gd(intial_w, lr, epochs)
    for epoch, W in enumerate(results):
        print(f'Epoch {epoch}: {W}')