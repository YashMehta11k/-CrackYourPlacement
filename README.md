### -CrackYourPlacement
45 Days Challenge to help you prepare for your upcoming DSA Interviews 


## Day-1

# 1. Remove duplicates from the sorted array - two pointer method
   - start from i=j=1
   - when the prev element is equal to the current element j pointer stays at the unique element and i moves forward
   - when prev element is not equal to the current element,we replace the duplicate places by a unique value
   - 
     int removeDuplicates(vector<int>& nums) {
        int j=1;
        for(int i=1;i<nums.size();i++){
            if(nums[i]!=nums[i-1]){
                nums[j]=nums[i];
                j++;
            }
        }
        return j;
    }

# 2. Moving all zeros to the right maintaining the order of non zero elements -two pointer
   - start from i=j=0
   - i moves forward till it find a non zero element
   - j remains there at the zero element position
   - swap elements at i and j when a non zero element is found
   - 
     void moveZeroes(vector<int>& nums) {
        int j=0;
        for(int i=0;i<nums.size();i++){
            if(nums[i]!=0){
                swap(nums[j],nums[i]);
                j++;
            }
        }
    }
  
# 3. Best time t buy and sell a stock - Kadane's algorithmn dynammic programming
   - buy is at index 0
   - update buy if a cheaper stock is found
   - update profit if the diff between the current price of the stock - buy is greater than the prev profit
   - 
     int maxProfit(vector<int>& prices) {
        int buy=prices[0],profit=0;
        for(int i=1;i<prices.size();i++){
            if(prices[i]<buy)
                buy=prices[i];
            else if(prices[i]-buy>profit)
                profit=prices[i]-buy;        
        }
        return profit;
    }
  
# 4. Two Sum - HashMap
   - if target - present num is present in the hashmap , return {i,pairIdx[target-num]}
   - pairIdx[num]=i
   -
     vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int,int> pairIdx;
        for(int i=0;i<nums.size();i++){
            int num=nums[i];
            if(pairIdx.find(target-num)!=pairIdx.end()){
                return {i,pairIdx[target-num]};
            }
            pairIdx[num]=i;
        }
        return {};
    }
  
# 5. Merge sorted array - two pointer method
  - intialise k=m+n-1
  - iterating from the end of both the arrays, find the largest ele from the end of both the arrays and initialise it to the end of array 1
  - 
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        int i=m-1,j=n-1,k=m+n-1;
        while(i>=0 && j>=0){
            if(nums1[i]>nums2[j]){
                nums1[k]=nums1[i];
                k--;i--;
            }else{
                nums1[k]=nums2[j];
                k--;j--;
            }
        }
        while(j>=0){
            nums1[k]=nums2[j];
            k--;j--;
        }
    }
  

## Day-2

# 1. Majority element - Moore's Voting Algorithm or Unordered map frequency
   - count the freq of each element
   - if map.second is greater than n/2 return it
   - 
      int majorityElement(vector<int>& nums) {
           unordered_map<int, int> mp; 
           int n = nums.size() / 2; 
           for (auto a : nums) {
               mp[a]++;
           }
           int ans = 0; 
           for (auto a : mp) {
               if (a.second > n) {
                   ans = a.first; 
               }
           }
           return ans;    
      }
  

# 2. Duplicate number - Slow and fast two pointer method
   - Traverse the array such that is slow pointer moves one step at a time while fast moves two steps at a time. This continues until slow and fast meet at the same position within the cycle.
   - After reinitialiszing slow at start, Again traverse the array such that slow and fast move one step at a time. They continue moving until they meet again at the start of the cycle as the distance between slow and fast is decreasing by one in each iteration.
   - 
     int findDuplicate(vector<int>& nums) {
        int slow=nums[0];
        int fast=nums[0];
        do{
            slow=nums[slow];
            fast=nums[nums[fast]];
        }while(slow!=fast);
        slow=nums[0];
        while(slow!=fast){
            slow=nums[slow];
            fast=nums[fast];
        }
        return slow;
    }


# 3. Sorting colors - 3 pointers until mid crosses high
   - If nums[mid] is 0 (red), swap it with nums[low], increment both low and mid.
   - If nums[mid] is 1 (white), leave it in place and increment mid.
   - If nums[mid] is 2 (blue), swap it with nums[high], decrement right.
   - 
     void sortColors(vector<int>& nums) {
        int low=0,high=nums.size()-1,mid=0;
        while(mid<=high){
            if(nums[mid]==0){
                swap(nums[low],nums[mid]);
                low++;
                mid++;
            }else if(nums[mid]==1){
                mid++;
            }else{
                swap(nums[mid],nums[high]);
                high--;
            }
        }
    }
  

# 4. Buy and Sell the stock part 2 - Greedy algorithm
   - iterate from index 1 and check if the current price is greater than the profit price then add the difference to the max.
   - Regardless of whether a profit was made or not, the start is updated to the current price (prices[i]). This step prepares for the next iteration, considering the current day's price as the new buying price.
   - 
     int maxProfit(vector<int>& prices) {
        int max=0;
        int start=prices[0];
        for(int i=1;i<prices.size();i++){
            if(start<prices[i]){
                max+=prices[i]-start;
            }
            start=prices[i];
        }
        return max;
    }

# 5. Find All the duplicates - Hashmap
   - doing -1 because length = n => index from 0 to n-1  and range of numbers = [1,n]
   - if value at the index is negative => already visited
   - if its not negative we mark it as negative as visited
   - 
     vector<int> findDuplicates(vector<int>& nums) {
        vector<int> ans;
        for(int i=0;i<nums.size();i++){
            int k=abs(nums[i])-1;
            if(nums[k]<0){
                ans.push_back(abs(nums[i]));
            }else{
                nums[k]*=-1;
            }
        }
        return ans;
    }
                                   

   
