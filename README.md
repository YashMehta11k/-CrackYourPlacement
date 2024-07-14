# -CrackYourPlacement
45 Days Challenge to help you prepare for your upcoming DSA Interviews 


# Day-1

1. Remove duplicates from the sorted array - two pointer method
   - start from i=j=1
   - when the prev element is equal to the current element j pointer stays at the unique element and i moves forward
   - when prev element is not equal to the current element , nums[j] = nums[i] i.e we replace the duplicate places by a unique value

2. Moving all zeros to the right maintaining the order of non zero elements -two pointer
   - start from i=j=0
   - i moves forward till it find a non zero element
   - j remains there at the zero element position
   - swap elements at i and j when a non zero element is found
  
3. Best time t buy and sell a stock - Kadane's algorithmn dynammic programming
   - buy is at index 0
   - update buy if a cheaper stock is found
   - update profit if the diff between the current price of the stock - buy is greater than the prev profit
  
4. Two Sum - HashMap
   - if target - present num is present in the hashmap , return {i,pairIdx[target-num]}
   - pairIdx[num]=i  
