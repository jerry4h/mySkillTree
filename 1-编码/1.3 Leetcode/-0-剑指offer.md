主要记录个人认为的重点题，作为总结。范围覆盖剑指offer、HOT100。
# 剑指offer
## 11 旋转数组中的最小数字
>把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。例如，数组 [3,4,5,1,2] 为 [1,2,3,4,5] 的一个旋转，该数组的最小值为1。  
示例 1：
输入：[3,4,5,1,2]
输出：1
示例 2：
输入：[2,2,2,0,1]
输出：0
来源：力扣（LeetCode）
链接：[https://leetcode-cn.com/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof](https://leetcode-cn.com/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof)

注意：得出的中间变量要记录其范围，分情况的切入点尽量清晰，参与判定的变量尽量少。
```python
class Solution:
    def minArray(self, numbers: List[int]) -> int:
        if not isinstance(numbers, list) or len(numbers) < 1:
            return
        i, j = 0, len(numbers)-1
        while(i < j):
            m = (i+j) >> 1  # i <= m < j
            if numbers[m] > numbers[j]: i = m + 1  # m位于左排序数组
            elif numbers[m] < numbers[j]: j = m  # m位于右排序数组
            else: j -= 1  # numbers[m] == numbers[j], 单独判断。
        return numbers[i]
```

## 12 矩阵中的路径

> 请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一格开始，每一步可以在矩阵中向左、右、上、下移动一格。如果一条路径经过了矩阵的某一格，那么该路径不能再次进入该格子。例如，在下面的3×4的矩阵中包含一条字符串“bfce”的路径（路径中的字母用加粗标出）。
[["a","b","c","e"],
["s","f","c","s"],
["a","d","e","e"]]
但矩阵中不包含字符串“abfb”的路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入这个格子。
示例 1：
输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
输出：true
示例 2：
输入：board = [["a","b"],["c","d"]], word = "abcd"
输出：false
来源：力扣（LeetCode）
链接：[https://leetcode-cn.com/problems/ju-zhen-zhong-de-lu-jing-lcof](https://leetcode-cn.com/problems/ju-zhen-zhong-de-lu-jing-lcof)

注意：dfs注意两点：恢复现场、执行判断（这里写在了dfs循环开头）
```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        if not isinstance(board, list) or not isinstance(word, str):
            return
        if len(board) == 0 or len(board[0]) == 0:
            return
        nrow, ncol = len(board), len(board[0])
        def dfs(i, j, k):  # 当前位置在i, j，从第k个word开始找，返回True或False。
            if k == len(word): return True
            elif not (0<=i<nrow and 0<=j<ncol) or board[i][j] != word[k]: return False
            temp, board[i][j] = board[i][j], None
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            for direct in directions:
                if dfs(i+direct[0], j+direct[1], k+1): return True
            board[i][j] = temp
            return False
        for i in range(nrow):
            for j in range(ncol):
                if dfs(i, j, 0): return True
        return False
```

## 14 切绳子
> 给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m] 。请问 k[0]*k[1]*...*k[m] 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。
答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。
示例 1：
输入: 2
输出: 1
解释: 2 = 1 + 1, 1 × 1 = 1
示例 2:
输入: 10
输出: 36
解释: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36
提示：
2 <= n <= 1000
来源：力扣（LeetCode）
链接：[https://leetcode-cn.com/problems/jian-sheng-zi-ii-lcof](https://leetcode-cn.com/problems/jian-sheng-zi-ii-lcof)

注意：推几个发现规律，可以证明(n-3)*3>n对n>=5都是成立的，然后对4的特殊情况讨论（余3得1的情况），结果取模（Python比较方便不会溢出，但C++可能不能直接求幂了，得循环相乘同时求余数）
```python
class Solution:
    def cuttingRope(self, n: int) -> int:
        if not isinstance(n, int) or n < 2:
            return
        MAX = 1000000007
        if n == 2:
            return 1
        elif n == 3:
            return 2
        i, j = n // 3, n % 3
        if j == 0:
            return 3**i % MAX
        elif j == 1:
            return 3**(i-1)*4 % MAX
        else:
            return 3**i*2 % MAX
```
## 16 *数值的整数次方
> 实现函数double Power(double base, int exponent)，求base的exponent次方。不得使用库函数，同时不需要考虑大数问题。
示例 1:
输入: 2.00000, 10
输出: 1024.00000
示例 2:
输入: 2.10000, 3
输出: 9.26100
示例 3:
输入: 2.00000, -2
输出: 0.25000
解释: 2-2 = 1/22 = 1/4 = 0.25
说明:
-100.0 < x < 100.0
n 是 32 位有符号整数，其数值范围是 [−231, 231 − 1] 。
来源：力扣（LeetCode）
链接：[https://leetcode-cn.com/problems/shu-zhi-de-zheng-shu-ci-fang-lcof](https://leetcode-cn.com/problems/shu-zhi-de-zheng-shu-ci-fang-lcof)

快速幂典型题。
```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if x is None or n is None:
            return 
        if x == 0:
            return 0  # 注意隐含非法条件
        if n < 0: x, n = 1/x, -n
        res = 1
        while(n>0):
            if n & 1:
                res *= x
            x *= x
            n >>= 1
        return res
```

## 19 正则表达式匹配 TODO
> 请实现一个函数用来匹配包含'. '和'*'的正则表达式。模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（含0次）。在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但与"aa.a"和"ab*a"均不匹配。
示例 1:
输入:
s = "aa"
p = "a"
输出: false
解释: "a" 无法匹配 "aa" 整个字符串。
示例 2:
输入:
s = "aa"
p = "a*"
输出: true
解释: 因为 '*' 代表可以匹配零个或多个前面的那一个元素, 在这里前面的元素就是 'a'。因此，字符串 "aa" 可被视为 'a' 重复了一次。
示例 3:
输入:
s = "ab"
p = ".*"
输出: true
解释: ".*" 表示可匹配零个或多个（'*'）任意字符（'.'）。
示例 4:
输入:
s = "aab"
p = "c*a*b"
输出: true
解释: 因为 '*' 表示零个或多个，这里 'c' 为 0 个, 'a' 被重复一次。因此可以匹配字符串 "aab"。
示例 5:
输入:
s = "mississippi"
p = "mis*is*p*."
输出: false
s 可能为空，且只包含从 a-z 的小写字母。
p 可能为空，且只包含从 a-z 的小写字母，以及字符 . 和 *。
来源：力扣（LeetCode）
链接：[https://leetcode-cn.com/problems/zheng-ze-biao-da-shi-pi-pei-lcof](https://leetcode-cn.com/problems/zheng-ze-biao-da-shi-pi-pei-lcof)

## 40 *最小的k个数
> 输入整数数组 arr ，找出其中最小的 k 个数。例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。
示例 1：
输入：arr = [3,2,1], k = 2
输出：[1,2] 或者 [2,1]
示例 2：
输入：arr = [0,1,2,1], k = 1
输出：[0]
限制：
0 <= k <= arr.length <= 10000
0 <= arr[i] <= 10000
来源：力扣（LeetCode）
链接：[https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof](https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof)

经典排序题。最大/小的k个数可以用排序来处理，堆排序时间复杂度O(NlogK)。如果用快速排序思想，时间复杂度就是O(NlogN)
快速排序的partition需要记忆。

```python
class Solution:
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        if not arr or k < 1:
            return []
        #思路1 快排 nlogn
        '''
        return sorted(arr)[:k]
        '''
        #思路2. 最小堆排序：自己实现堆排序插入操作太复杂，不推荐
        heap = self.build_heap(arr[:k])
        for i in range(k, len(arr)):
            if arr[i] < heap[0]:
                heap[0] = arr[i]
                self.sink(heap, 0)
        return heap
    def build_heap(self, nums):
        heap = nums
        n = len(heap)
        for i in range(n//2, -1, -1):
            self.sink(heap, i)
        return heap
    def sink(self, heap, k):
        largest = k
        l = 2 * k
        r = 2 * k + 1
        if l < len(heap) and heap[l] > heap[largest]:
            largest = l
        if r < len(heap) and heap[r] > heap[largest]:
            largest = r
        if largest != k:
            heap[k], heap[largest] = heap[largest], heap[k]
            self.sink(heap, largest)
        #思路3. 快速排序思想
        # 如果pivot位置就是k，则break返回
        #    随机选取pivot，剩下快排
        #    判断下次位置，继续快排
        '''
        start, end = 0, len(arr)-1
        pivot = self.partition(arr, start, end)
        while(pivot!=k-1):
            if pivot > k-1:
                end = pivot - 1  # 这里加减格外要注意
                pivot = self.partition(arr, start, end)
            elif pivot < k-1:
                start = pivot + 1  # 这里加减格外要注意
                pivot = self.partition(arr, start, end)
        return arr[:k]
    def partition(self, arr, start, end):  # 这份经典partition代码要记忆。
        i = start - 1
        pivot = arr[end]
        for j in range(start, end):  # j在扫雷
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i+1], arr[end] = arr[end], arr[i+1]
        return i+1
    '''
```


## 41 数据流中的中位数

> 如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。
例如，
[2,3,4] 的中位数是 3
[2,3] 的中位数是 (2 + 3) / 2 = 2.5
设计一个支持以下两种操作的数据结构：
void addNum(int num) - 从数据流中添加一个整数到数据结构中。
double findMedian() - 返回目前所有元素的中位数。
示例 1：
输入：
["MedianFinder","addNum","addNum","findMedian","addNum","findMedian"]
[[],[1],[2],[],[3],[]]
输出：[null,null,null,1.50000,null,2.00000]
示例 2：
输入：
["MedianFinder","addNum","findMedian","addNum","findMedian"]
[[],[2],[],[3],[]]
输出：[null,null,2.00000,null,2.50000]
限制：
最多会对 addNum、findMedia进行 50000 次调用。
来源：力扣（LeetCode）
链接：[https://leetcode-cn.com/problems/shu-ju-liu-zhong-de-zhong-wei-shu-lcof](https://leetcode-cn.com/problems/shu-ju-liu-zhong-de-zhong-wei-shu-lcof)

思考：大顶堆和小顶堆的应用，保证小顶堆长度比大顶堆大一个或相等。由于python只有小顶堆，大顶堆的排序需要用负数来维持。
小顶堆：最小值在第一位，后续都比最小值要大。

```python
class MedianFinder:
    def __init__(self):
        """
        initialize your data structure here.
        """
        # 初始化大顶堆和小顶堆
        self.max_heap = []
        self.min_heap = []  # python的heapq默认为小顶堆，也好理解，排序算法默认为小到大排序。
    def addNum(self, num: int) -> None:
        if len(self.max_heap) == len(self.min_heap):# 先加到大顶堆，再把大堆顶元素加到小顶堆
            heapq.heappush(self.min_heap, -heapq.heappushpop(self.max_heap, -num))
        else:  # 先加到小顶堆，再把小堆顶元素加到大顶堆
            heapq.heappush(self.max_heap, -heapq.heappushpop(self.min_heap, num))
    def findMedian(self) -> float:
        if len(self.min_heap) == len(self.max_heap):
            return (-self.max_heap[0] + self.min_heap[0]) / 2
        else:
            return self.min_heap[0]
```

## 43 *1~n中整数1出现的次数
> 输入一个整数 n ，求1～n这n个整数的十进制表示中1出现的次数。
例如，输入12，1～12这些整数中包含1 的数字有1、10、11和12，1一共出现了5次。
 示例 1： 
 输入：n = 12
 输出：5 
 示例 2： 
 输入：n = 13 
 输出：6 
 限制： 1 <= n < 2^31
来源：力扣（LeetCode）
链接：[https://leetcode-cn.com/problems/1nzheng-shu-zhong-1chu-xian-de-ci-shu-lcof](https://leetcode-cn.com/problems/1nzheng-shu-zhong-1chu-xian-de-ci-shu-lcof/)

这题困扰了4个小时左右。难点在思路确定、分情况讨论、数值计算。还是比较难的一题。
**思路**：按照数值范围计算；按照位计算。两个思路都是可行的，实际考虑的时候都得考虑。我采用的是按照位运算的方法：对个、十、百...位单独计算1可能的出现次数。
**分情况讨论**：考虑的话得从两个维度：位上的数和1的关系、位上的数在左右边界情况（可以先考虑通用情况，因为左右边界条件是中间情况的特例。我第一遍因为计算错误卡了很久。）
**数值计算**：这里数值计算涉及到数数，也是细节和一大难点。具体思路有两个：排列组合思路、大减小思路。
[参考解析](https://leetcode-cn.com/problems/1nzheng-shu-zhong-1chu-xian-de-ci-shu-lcof/solution/wo-de-jian-dan-jie-fa-by-ai-bian-cheng-de-zhou-n-2/)
```python
import math
class Solution:
    def countDigitOne(self, n: int) -> int:
        if not isinstance(n, int) or n < 1:
            return 0   
       
        strn = str(n)
        cnt = 0
        for idx, s in enumerate(strn):
            num = int(s)
            left = int(strn[:idx]) if idx>=1 else 0
            right = int(strn[idx+1:]) if idx<=len(strn)-2 else 0
            nr = len(strn) - idx-1
            if num > 1:
                cnt += (left+1)*10**nr
            elif num == 1:
                cnt += left*10**nr + (right+1)
            else:
                cnt += left*10**nr
        return cnt
```

## 44 *数字序列中某一位的数字

> 数字以0123456789101112131415…的格式序列化到一个字符序列中。在这个序列中，第5位（从下标0开始计数）是5，第13位是1，第19位是4，等等。
请写一个函数，求任意第n位对应的数字。
示例 1：
输入：n = 3
输出：3
示例 2：
输入：n = 11
输出：0
限制：
0 <= n < 2^31
来源：力扣（LeetCode）
链接：[https://leetcode-cn.com/problems/shu-zi-xu-lie-zhong-mou-yi-wei-de-shu-zi-lcof](https://leetcode-cn.com/problems/shu-zi-xu-lie-zhong-mou-yi-wei-de-shu-zi-lcof)

同样困扰了我2小时左右，难点在于思路、数值计算、边界条件、索引和数数的数值理解。
**思路有两个**：模拟法、分阶段计算；n也有数值溢出的风险。模拟法会时间溢出，所以得分阶段计算。
**分阶段计算**：先确定目标的位数digit；再确定目标数num；最后确定索引index。
**特别总结**：1. 模拟法写循环注意提前规定循环体外的变量含义，这样就不会糊涂；2. 数值有两层含义：索引、差值。加减一个数m，代表两个索引差m步，说明现在至少一个(m+1)个数；3. 跟数值、索引打交道很容易弄混淆。
**TODO：这题改日还得重新写，下面代码为复制题解里的。感觉作者理解很深。** [题解链接](https://leetcode-cn.com/problems/shu-zi-xu-lie-zhong-mou-yi-wei-de-shu-zi-lcof/solution/zhe-shi-yi-dao-shu-xue-ti-ge-zhao-gui-lu-by-z1m/)

```python
class Solution:
    def findNthDigit(self, n: int) -> int:
        '''这是模拟的方法。循环要不弄糊涂，需要规定循环结束前后的含义。
        m = 0
        while(n>=len(str(m))):
            n -= len(str(m))
            m += 1
        return int(str(m)[n])
        '''
        n -= 1
        for digits in range(1, 11):
            first_num = 10**(digits - 1)
            if n < 9 * first_num * digits:
                return int(str(first_num + n/digits)[n%digits])
            n -= 9 * first_num * digits
```
## 45 *把数组排成最小的数
> 输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。
示例 1:
输入: [10,2]
输出: "102"
示例 2:
输入: [3,30,34,5,9]
输出: "3033459"
提示:
0 < nums.length <= 100
来源：力扣（LeetCode）
链接：[https://leetcode-cn.com/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof](https://leetcode-cn.com/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof)

技巧在于python内对sorted函数的利用、字符串内置比较的利用。
通过传入比较的方式key属性，达到利用内置排序库的目的。
注意`functools.cmp_to_key`的用法、`''.join()`的用法。
**TODO：** compare1和compare2用作key的结果不一样，如下：
```python
compare1 = lambda x, y: x+y > y+x
compare2 = lambda x, y: 1 if x+y > y+x else -1
# 比较下compare1和compare2，用在key的区别。用compare1的话会逻辑出错。
```
[题解链接](https://leetcode-cn.com/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/comments/246676)
```python
class Solution:
    def minNumber(self, nums: List[int]) -> str:
        from functools import cmp_to_key
        # def compare(a,b):
        #     return 1 if a+b > b+a else -1
        compare = lambda a, b: 1 if a+b>b+a else -1
        nums = sorted([str(i) for i in nums], key=cmp_to_key(compare))
        return ''.join(nums)
```

## 46 把数字翻译成字符串
> 给定一个数字，我们按照如下规则把它翻译为字符串：0 翻译成 “a” ，1 翻译成 “b”，……，11 翻译成 “l”，……，25 翻译成 “z”。一个数字可能有多个翻译。请编程实现一个函数，用来计算一个数字有多少种不同的翻译方法。
示例 1:
输入: 12258
输出: 5
解释: 12258有5种不同的翻译，分别是"bccfi", "bwfi", "bczi", "mcfi"和"mzi"
 提示
0 <= num < 231
来源：力扣（LeetCode）
链接：[https://leetcode-cn.com/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof](https://leetcode-cn.com/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof)

典型的dfs和dp的应用，可以当做有条件的青蛙跳台阶问题。
**小坑**：两数字合并的条件是10~25，因为类似06是不合法的。
**TODO** 下面代码里的cnt为什么必须设为类属性了？

```python
class Solution:
    def translateNum(self, num: int) -> int:
        '''
        self.cnt = 0
        def dfs(res):
            if len(res) == 0:
                self.cnt += 1 # 为何这里不能直接用cnt的局部变量，必须要self?之前就可以用来着。
                return
            dfs(res[1:])
            if len(res)>=2 and 10<=int(res[:2])<=25:
                dfs(res[2:])
        dfs(str(num))
        return self.cnt
        '''
        # dp方法。
        if not isinstance(num, int) or num < 0:
            return 0
        strn = str(num)
        dp = [0]*len(strn) # dp[i]表示包含索引i的之前元素组成的字符串个数
        dp[0] = 1
        if len(strn) == 1:
            return dp[0]
        dp[1] = 2 if 10<=int(strn[:2])<=25 else 1
        if len(strn) >= 3:
            i = 2
            while(i<len(strn)):
                if 10<=int(strn[i-1:i+1])<=25:
                    dp[i] = dp[i-2] + dp[i-1]
                else:
                    dp[i] = dp[i-1]
                i += 1
        return dp[-1]
```
## 47 礼物的最大价值

> 在一个 m*n 的棋盘的每一格都放有一个礼物，每个礼物都有一定的价值（价值大于 0）。你可以从棋盘的左上角开始拿格子里的礼物，并每次向右或者向下移动一格、直到到达棋盘的右下角。给定一个棋盘及其上面的礼物的价值，请计算你最多能拿到多少价值的礼物？
示例 1:
输入: 
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
输出: 12
解释: 路径 1→3→5→2→1 可以拿到最多价值的礼物
来源：力扣（LeetCode）
链接：[https://leetcode-cn.com/problems/li-wu-de-zui-da-jie-zhi-lcof](https://leetcode-cn.com/problems/li-wu-de-zui-da-jie-zhi-lcof)

**说明**：dfs思路和dp思路。很明显dp更简洁。
**小坑**：一开始老想着按照对角线计算，写的很复杂，而且没必要。完全可以按照每一行计算。
**反思**：思考的方式一定要先横向思考。尤其在动手写代码之前。
```python
class Solution:
    def maxValue(self, grid: List[List[int]]) -> int:
        if not isinstance(grid, list) or len(grid) < 1 or len(grid[0]) < 1:
            return 0
        m, n = len(grid), len(grid[0])
        dp = [0]*n  # 表示反复利用的dp空间
        for i in range(m):
            for j in range(n):
                left = dp[j-1] if j>0 else 0
                dp[j] = grid[i][j] + max(dp[j], left)
        return dp[n-1]
```

## 48 *最长不含重复字符的子字符串

> 请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度。
示例 1:
输入: "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
示例 2:
输入: "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
示例 3:
输入: "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
来源：力扣（LeetCode）
链接：[https://leetcode-cn.com/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof](https://leetcode-cn.com/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof)

**思路**：同样是dp的思路，计算的仍然是以index结尾时，满足条件的结果。
**小坑**：涉及到字符查找的技巧，利用字典存储字符的index。
**反思**：利用字典存储索引，是值得记忆的小技巧。
```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if not isinstance(s, str) or len(s) == 0:
            return 0
        dic = {}
        cur_len = 0
        max_len = 0
        for i, char in enumerate(s):
            if char not in dic or i-dic[char]>cur_len:
                cur_len += 1
            else:
                cur_len = i-dic[char]
            dic[char] = i
            if cur_len > max_len:
                max_len = cur_len
        return max_len
```

## 49 *丑数 

> 我们把只包含因子 2、3 和 5 的数称作丑数（Ugly Number）。求按从小到大的顺序的第 n 个丑数。
示例:
输入: n = 10
输出: 12
解释: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12 是前 10 个丑数。
说明:  
1 是丑数。
n 不超过1690。
来源：力扣（LeetCode）
链接： [https://leetcode-cn.com/problems/chou-shu-lcof](https://leetcode-cn.com/problems/chou-shu-lcof)

**思路**：这题是动态规划dp的变形，难度还挺大。反向思维，首先，某个数m能拆成``n * (2/3/5)``，那么n同样也会出现在丑数的列表里；其次，n的出现是有顺序的，对2/3/5最小值的出现，是来自对丑数的按序遍历。因此，2/3/5分别定义一个指针，对丑数序列遍历。
**小坑**：如代码注释，每次计算2/3/5可能出现重复数值，如果直接取``outs.index(num)``，恭喜入坑，而且比较隐蔽。
**反思**：其一，本题用找规律很难找，因为乘法相乘，每组有规律的组数长度是增加的，难确定每组的长度；其二，后续应当总结一下**代码容易出错的点**，求``min/max``函数和取index值就算一个，常见的还有：索引与数目计算、各种边界条件（循环条件判断、判断加不加等号）、递归现场还原、忘记计数等等；其三，**尽量固定资源需求**，用for循环，少用while；列表长度一开始就初始化好，少用append()

```python
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        nums = [0]*n  #还是申请好资源。append速度太慢了。
        nums[0] = 1
        p = [0, 0, 0]
        for i in range(1, n):
            outs = [nums[p[0]]*2, nums[p[1]]*3, nums[p[2]]*5] # 可能会有重复的。。。
            num = min(outs)
            for j in range(3):
                if outs[j] == num: p[j] += 1
            nums[i] = num
        return nums[n-1]
```
## 51 *数组中的逆序对

> 在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数。
示例 1:
输入: [7,5,6,4]
输出: 5
限制：
0 <= 数组长度 <= 50000
在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数。
来源：力扣（LeetCode）
链接：[https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof](https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof)

思路：模拟的方法时间复杂度O(n^2)，分治思想归并排序时间复杂度O(nlogn)，空间复杂度O(n)
小坑：注意归并排序的写法，先保证左右分治，再进行归并排序。
```python
class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        if not isinstance(nums, list) or len(nums) == 0:
            return 0
        self.cnt = 0
        l, r = 0, len(nums)-1
        nums = self.reversePairsCore(nums, l, r)
        return self.cnt

    def reversePairsCore(self, nums, i, j):
        if i>=j:
            return nums
        mid = (i+j) >> 1
        nums = self.reversePairsCore(nums, i, mid)
        nums = self.reversePairsCore(nums, mid+1, j)
        p1, p2 = i, mid+1  # P1容易写错，涉及多个变量一定要画图。
        pattern = [0]*(j-i+1)
        k = 0
        while(p1<=mid and p2 <= j):
            if nums[p2]<nums[p1]:
                self.cnt += mid-p1+1
                pattern[k] = nums[p2]
                p2 += 1
            else:
                pattern[k] = nums[p1]
                p1 += 1
            k += 1
        while(p1<=mid):
            pattern[k] = nums[p1]
            p1 += 1
            k += 1
        while(p2<=j):
            pattern[k] = nums[p2]
            p2 += 1
            k += 1
        nums[i:j+1] = pattern
        return nums
```

## 56-I 数组中数字出现的次数

> 一个整型数组 nums 里除两个数字之外，其他数字都出现了两次。请写程序找出这两个只出现一次的数字。要求时间复杂度是O(n)，空间复杂度是O(1)。
示例 1：
输入：nums = [4,1,4,6]
输出：[1,6] 或 [6,1]
示例 2：
输入：nums = [1,2,10,4,1,4,3,3]
输出：[2,10] 或 [10,2]
限制：
2 <= nums <= 10000
来源：力扣（LeetCode）
链接：[https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof](https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof)

```python
class Solution:
    def singleNumbers(self, nums: List[int]) -> List[int]:
        ret = 0  # 0与任何数的异或都为本身
        a, b = 0, 0
        for num in nums:
            ret ^= num
        i = 1
        while(i & ret == 0):  # 随便找到一位不为0的位置
            i <<= 1
        for num in nums:
            if num & i == 0:
                a ^= num
            else:
                b ^= num
        return [a, b]
```

## 56-II 数组中数字出现的次数II
> 在一个数组 nums 中除一个数字只出现一次之外，其他数字都出现了三次。请找出那个只出现一次的数字。
示例 1：
输入：nums = [3,4,3,3]
输出：4
示例 2：
输入：nums = [9,1,7,9,7,9,7]
输出：1
限制：
1 <= nums.length <= 10000
1 <= nums[i] < 2^31
来源：力扣（LeetCode）
链接：[https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-ii-lcof](https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-ii-lcof)

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        if not isinstance(nums, list) or len(nums) == 0:
            return
        '''位运算
        res = 0
        for i in range(32):
            prob = 1 << i
            cnt = 0
            for num in nums:
                if num & prob > 0:
                    cnt += 1
            if cnt % 3 != 0:
                res |= prob
        return res
        '''
        dic = {}
        for num in nums:
            if num not in dic:
                dic[num] = 0
            dic[num] += 1
            if dic[num] == 3:
                dic.pop(num)
        res = list(dic.keys())[0]
        return res
```


## 59 队列的最大值
> 请定义一个队列并实现函数 max_value 得到队列里的最大值，要求函数max_value、push_back 和 pop_front 的均摊时间复杂度都是O(1)。
若队列为空，pop_front 和 max_value 需要返回 -1
示例 1：
输入: 
["MaxQueue","push_back","push_back","max_value","pop_front","max_value"]
[[],[1],[2],[],[],[]]
输出: [null,null,null,2,1,2]
示例 2：
输入: 
["MaxQueue","pop_front","max_value"]
[[],[],[]]
输出: [null,-1,-1]
限制：
1 <= push_back,pop_front,max_value的总操作数 <= 10000
1 <= value <= 10^5
来源：力扣（LeetCode）
链接：[https://leetcode-cn.com/problems/dui-lie-de-zui-da-zhi-lcof](https://leetcode-cn.com/problems/dui-lie-de-zui-da-zhi-lcof)

```python
class MaxQueue:
    def __init__(self):
        from collections import deque
        self.queue = deque()  # 双端队列速度更快。
        self.max_queue = deque()  # 最大值（可以重复）
    def max_value(self) -> int:
        if len(self.max_queue)== 0:
            return -1
        return self.max_queue[0]
    def push_back(self, value: int) -> None:
        self.queue.append(value)
        i = len(self.max_queue)-1
        while(i>=0 and self.max_queue[i]<value):  # 取小于号，将最大值保留下来
            self.max_queue.pop()
            i -= 1
        self.max_queue.append(value)
    def pop_front(self) -> int:
        if len(self.queue) == 0:
            return -1
        value = self.queue.popleft()
        if value == self.max_queue[0]:
            self.max_queue.popleft()
        return value
```

## 63 股票的最大利润
> 假设把某股票的价格按照时间先后顺序存储在数组中，请问买卖该股票一次可能获得的最大利润是多少？
示例 1:
输入: [7,1,5,3,6,4]
输出: 5
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格。
示例 2:
输入: [7,6,4,3,1]
输出: 0
解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。
限制：
0 <= 数组长度 <= 10^5
来源：力扣（LeetCode）
链接：[https://leetcode-cn.com/problems/gu-piao-de-zui-da-li-run-lcof](https://leetcode-cn.com/problems/gu-piao-de-zui-da-li-run-lcof)

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not isinstance(prices, list) or len(prices) < 2:
            return 0
        dpi = 0  # 表示当前第i个数可能的最大利润
        minpri = prices[0]  # 表示前i个数（包括i）的最低价格。
        for i in range(1, len(prices)):
            dpi = max(dpi, prices[i]-minpri)
            minpri = min(minpri, prices[i])
        return dpi
```

## 64 求1+2+...+n
> 求 1+2+...+n ，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。
示例 1：
输入: n = 3
输出: 6
示例 2：
输入: n = 9
输出: 45
限制：
1 <= n <= 10000
来源：力扣（LeetCode）
链接：[https://leetcode-cn.com/problems/qiu-12n-lcof](https://leetcode-cn.com/problems/qiu-12n-lcof)

```python
class Solution:
    def sumNums(self, n: int) -> int:
        return n and n+self.sumNums(n-1)  # and 和 or 并非返回False或True，而是返回判断的最后一步
```

## 67 把字符串转成整数
> 写一个函数 StrToInt，实现把字符串转换成整数这个功能。不能使用 atoi 或者其他类似的库函数。
首先，该函数会根据需要丢弃无用的开头空格字符，直到寻找到第一个非空格的字符为止。
当我们寻找到的第一个非空字符为正或者负号时，则将该符号与之后面尽可能多的连续数字组合起来，作为该整数的正负号；假如第一个非空字符是数字，则直接将其与之后连续的数字字符组合起来，形成整数。
该字符串除了有效的整数部分之后也可能会存在多余的字符，这些字符可以被忽略，它们对于函数不应该造成影响。
注意：假如该字符串中的第一个非空格字符不是一个有效整数字符、字符串为空或字符串仅包含空白字符时，则你的函数不需要进行转换。
在任何情况下，若函数不能进行有效的转换时，请返回 0。
说明：
假设我们的环境只能存储 32 位大小的有符号整数，那么其数值范围为 [−231,  231 − 1]。如果数值超过这个范围，请返回  INT_MAX (231 − 1) 或 INT_MIN (−231) 。
示例 1:
输入: "42"
输出: 42
示例 2:
输入: "   -42"
输出: -42
解释: 第一个非空白字符为 '-', 它是一个负号。
     我们尽可能将负号与后面所有连续出现的数字组合起来，最后得到 -42 。
示例 3:
输入: "4193 with words"
输出: 4193
解释: 转换截止于数字 '3' ，因为它的下一个字符不为数字。
示例 4:
输入: "words and 987"
输出: 0
解释: 第一个非空字符是 'w', 但它不是数字或正、负号。
     因此无法执行有效的转换。
示例 5:
输入: "-91283472332"
输出: -2147483648
解释: 数字 "-91283472332" 超过 32 位有符号整数范围。 
     因此返回 INT_MIN (−231) 。
来源：力扣（LeetCode）
链接：[https://leetcode-cn.com/problems/ba-zi-fu-chuan-zhuan-huan-cheng-zheng-shu-lcof](https://leetcode-cn.com/problems/ba-zi-fu-chuan-zhuan-huan-cheng-zheng-shu-lcof)

```python
class Solution:
    def strToInt(self, str: str) -> int:
        INT_MAX = 2147483647
        INT_MIN = -2147483648
        s= str.strip()  # 返回一个拷贝，把两边的空格都去掉。
        res = re.compile(r'^[\+\-]?\d+')  # ^匹配起点；[\+\-]?可能出现正负；\d数字相当于[0-9]；\d+匹配1到多个
        num = res.findall(s)
        nums = int(*num)
        return max(min(nums, INT_MAX), INT_MIN)
```

