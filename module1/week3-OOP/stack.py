class Stack:
    def __init__(self, capacity):
        self.capacity = capacity
        self.items = []

    def is_empty(self):
        return len(self.items) == 0

    def is_full(self):
        return len(self.items) == self.capacity

    def pop(self):
        if self.is_empty():
            raise IndexError('pop from empty stack')
        return self.items.pop()

    def push(self, value):
        if self.is_full():
            raise IndexError('push to full stack')
        self.items.append(value)

    def top(self):
        if self.is_empty():
            raise IndexError('top from empty stack')
        return self.items[-1]


if __name__ == '__main__':
    stack1 = Stack(capacity=5)

    stack1.push(1)
    assert stack1.is_full() == False
    stack1.push(2)
    print(stack1.is_full())
    print(stack1.top())
    print(stack1.pop())
    print(stack1.top())
    print(stack1.pop())
    print(stack1.is_empty())
    # print(stack1.pop())

    for i in range(stack1.capacity + 1):
        stack1.push(i)
