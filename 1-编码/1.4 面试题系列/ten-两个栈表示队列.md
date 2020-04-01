# 两个栈实现队列

代码逻辑好写，对于边界条件、提示情况需要着重考虑。

```Python
import pdb
class stackQueue:
    def __init__(self):
        self.stack1 = []
        self.stack2 = []
    def dump(self, x):
        if not x: return  # 边界条件没有考虑
        self.stack1.append(x)
    def pop(self):
        if self.stack2:
            return self.stack2.pop()
        else:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
            if self.stack2:  # 边界条件没有考虑
                return self.stack2.pop()
        return  # 这里要给提示
if __name__ == '__main__':
    sq = stackQueue()
    pdb.set_trace()
```