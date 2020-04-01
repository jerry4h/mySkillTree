
> 笔试题1：有n个组员时，有多少种组队方法？任选几个人出来成一队，每个队伍选择其中一个人担任组长。


这题硬算的话复杂度是`N^2`，但如果把Cm^n写作递推的关系的话，相当于增加状态变量，减少了Cm^n的计算复杂度。   
前面一直在考虑Dp的解法，想从n-1个组员的结果推到n个的结果...耽误了几乎半个小时的时间。
```Python
import sys 
for line in sys.stdin:
    a = line.rstrip()
    m = int(a)
    n = 1
    Cmn = m  # n = 1 时
    sums = Cmn
    while(n < m):
        n += 1
        Cmn = Cmn * (m-n+1) / (n)  # 不冷静下想不出来这种递推
        sums += n * Cmn
        sums %= 1000000007
    print(int(sums))
```