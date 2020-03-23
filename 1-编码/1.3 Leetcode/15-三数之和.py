class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        # 这题本身是N3的复杂度
        # 降为2维的夹逼法则：排序nlogn + 选取和n * 夹逼n = n2
        # 去重是这题的关键：重复数字和重复组合
        if len(nums) < 3:
            return []
        res = []
        nums.sort()
        for i, c in enumerate(nums):
            if i>0 and nums[i-1] == nums[i]:  # i的去重（重复的数字）
                continue
            j, k = i+1, len(nums)-1  # j=i+1 去重（重复的组合）
            while(j < k):
                if i == j:
                    j += 1
                    continue
                if i == k:
                    k -= 1
                    continue
                if nums[j] + nums[k] == -nums[i]:
                    res.append([nums[i], nums[j], nums[k]])
                    while(j<k and nums[j]==nums[j+1]):  # 去重是个技术活。j<k表示能j+1;j+=1和k-=1要注意（重复的数字）
                        j += 1
                    while(j<k and nums[k]==nums[k-1]):
                        k -= 1
                    j += 1
                    k -= 1
                elif nums[j] + nums[k] < -nums[i]:
                    j += 1
                else:
                    k -= 1
        return res  # 做完一定再看一遍题干，检查输入边界