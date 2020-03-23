class Solution:
    def maxArea(self, height: List[int]) -> int:
        # 这题硬解法是N2复杂度
        # 怎么做到Nlogn或者N的时间复杂度就好了

        # 问题在于求width*height的可能最大值
        # width = j-i, height = min(lst[i], lst[j])
        # 这种一般双指针或者动态规划之类的方法，这里双指针要理解，为什么能排除到其中一边的所有可能情况。
        # 得从一开始往后看。小的那一边i(左边)，肯定是可以忽略了，不可能有以它开头的更大面积的可能性了。
        # 所以后续的j（右边），不需要考虑它作为右边的最大值的可能性，因为左边的所有开头的都已经被否定了。
        if len(height) < 2:
            return 0
        i, j = 0, len(height)-1
        maxA = 0
        while(i<j):
            area = (j-i) * min(height[i], height[j])
            if area > maxA:
                maxA = area
            if height[i] > height[j]:
                j -= 1
            else:
                i += 1
        return maxA