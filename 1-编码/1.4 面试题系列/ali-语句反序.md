>  评测题目: 无  
> give the “this is the string”, write code to make this to “string the is this” 

这是由于笔试做的不好，面试官先给我出了道简单的题。

```Python
def reverseString(s):
    return s[::-1]
  
def reverse(s):
    if not s:
        return ''
    s = reverseString(s)
    return ' '.join([reverseString(string) for string in s.split(' ')])
```