class Queue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.items = []

    def is_empty(self):
        return len(self.items) == 0

    def is_full(self):
        return len(self.items) == self.capacity

    def dequeue(self):
        if self.is_empty():
            raise IndexError('dequeue from empty queue')
        return self.items.pop(0)

    def enqueue(self, value):
        if self.is_full():
            raise IndexError('enqueue to full queue')
        self.items.append(value)

    def front(self):
        if self.is_empty():
            raise IndexError('front from empty queue')
        return self.items[0]


if __name__ == '__main__':
    queue1 = Queue(capacity=5)

    queue1.enqueue(1)
    assert queue1.is_full() == False
    queue1.enqueue(2)
    print(queue1.is_full())
    print(queue1.front())
    print(queue1.dequeue())
    print(queue1.front())
    print(queue1.dequeue())
    print(queue1.is_empty())
    # print(queue1.dequeue())

    for i in range(queue1.capacity + 1):
        queue1.enqueue(i)
