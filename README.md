# -CrackYourPlacement
45 Days Challenge to help you prepare for your upcoming DSA Interviews 


## Day-1

### 1. Remove duplicates from the sorted array - two pointer method
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

### 2. Moving all zeros to the right maintaining the order of non zero elements -two pointer
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
  
### 3. Best time t buy and sell a stock - Kadane's algorithmn dynammic programming
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
  
### 4. Two Sum - HashMap
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
  
### 5. Merge sorted array - two pointer method
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

### 1. Majority element - Moore's Voting Algorithm or Unordered map frequency
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
  

### 2. Duplicate number - Slow and fast two pointer method
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


### 3. Sorting colors - 3 pointers until mid crosses high
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
  

### 4. Buy and Sell the stock part 2 - Greedy algorithm
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

### 5. Find All the duplicates - Hashmap
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


## Day-3

### 1. Set matrices as zero - Matrices and 
   - initialise a vector map to save the coordinates of ele which are 0
   - make all the rows and col as zero for each position
   -
         void setZeroes(vector<vector<int>>& matrix) {
                 int rows=matrix.size(),cols=matrix[0].size();
                 vector<pair<int,int>> mp;
                 for(int i=0;i<rows;i++){
                     for(int j=0;j<cols;j++){
                         if(matrix[i][j]==0)
                             mp.push_back(make_pair(i,j));
                     }
                 }
                 for(auto i:mp){
                     int j=0;
                     while(j<cols){
                         matrix[i.first][j]=0;
                         j++;
                     }
                     j=0;
                     while(j<rows){
                         matrix[j][i.second]=0;
                         j++;
                     }
                 }
             }

### 2. Choclate distribution - sliding window
   - sort the array and traverse to each window of size of number of childer and find the diff
   -
           long long findMinDiff(vector<long long> a, long long n, long long m){
              if (m == 0 || n == 0)
                  return 0;
              sort(a.begin(), a.end());
              if (n < m)
                  return -1;
              long long min_diff = LLONG_MAX;
              for (long long i = 0; i + m - 1 < n; ++i) {
                  long long diff = a[i + m - 1] - a[i];
                  if (diff < min_diff)
                      min_diff = diff;
              }
              return min_diff;
          } 

### 3. Subarray sums divided by k - Prefix sum with Hashmap
   - If it has, it means there are subarrays that sum to a multiple of ( k ), and you increase your count by how many times this remainder has been seen.
   - Update the hash map to include this new remainder.
   -
            int subarraysDivByK(vector<int>& nums, int k) {
              int count=0;
              int prefixSum=0;
              unordered_map<int,int> prefixMap;
              prefixMap[0]=1;
              for(int num :nums){
                  prefixSum+=num;
                  int mod=prefixSum%k;
                  if(mod<0){
                      mod+=k;
                  }
                  if(prefixMap.find(mod)!=prefixMap.end()){
                      count+=prefixMap[mod];
                      prefixMap[mod]+=1;
                  }else{
                      prefixMap[mod]=1;
                  }
              }
              return count;
          }
    
### 4. Container with most water - two pointers
   - intialise two pointers left and right at the ends of the array , call the max area at each window
   - always choose the larger height and shorter the window accordingly
   -
           int maxArea(vector<int>& height) {
              int left=0,right=height.size()-1;
              int area=0;
              while(left<right){
                  int curr=(right-left)*min(height[left],height[right]);
                  area=max(area,curr);
                  if(height[left]<height[right])
                      left++;
                  else
                      right--;
              }
              return area;
          }

### 5. Three Sum - two pointers
   - intialise the fixed element tar and find two numbers for the the sum =-tar
   - for each i , j=i+1 , and search in the window from j to k , according to the sum and tar
   - skip the repeating numbers
   -
           vector<vector<int>> threeSum(vector<int>& nums) {
              sort(nums.begin(),nums.end());
              vector<vector<int>> v;
              int sum=0;
              int tar=0;
              int j=0;
              int k=nums.size()-1;
              for(int i=0;i<nums.size()-2;i++){
                  if(i>0 && nums[i]==nums[i-1]) 
                      continue;
                  j=i+1;
                  k=nums.size()-1;
                  tar=-nums[i];
                  while(j<k){
                      sum=nums[j]+nums[k];
                      if(sum<tar){
                          j++;
                      }
                      else if(tar<sum){
                          k--;
                      }
                      else{
                          v.push_back({nums[i],nums[j],nums[k]});
                          j++;
                          k--;
                          while(j < k && nums[j] == nums[j - 1]) j++;
                          while(j < k && nums[k] == nums[k + 1]) k--;
                      }
                  }
              }
              return v;
          }

### 6. Four Sum
   - same as above with extra loop for j
   -
           vector<vector<int>> fourSum(vector<int>& nums, int target) {
              int n = nums.size();
              sort(nums.begin(), nums.end());
              vector<vector<int>> ans;
              for(int i=0; i<n-3; i++){
                  if(i>0 && nums[i] == nums[i-1])
                      continue;
                  for(int j = i+1; j<n-2; j++){
                      if(j>i+1 && nums[j]==nums[j-1])
                          continue;
                      int start = j+1, end = n-1;
                      long tempTarget = (long)target - (nums[i]+nums[j]);
                      while(start < end){
                          long tempSum = nums[end]+nums[start];
                          if(tempSum < tempTarget){
                              start++;
                          }else if(tempSum > tempTarget){
                              end--;
                          }else{
                              ans.push_back( {nums[i], nums[j], nums[start], nums[end] });
                              while(start < end && nums[end] == nums[end-1])
                                  end--;
                              while(start < end && nums[start] == nums[start+1])
                                  start++;
                              start++; end--;
                          }
                      }
                  }
              }
              return ans;
          }


## Day - 4

### 1. Maximum points obtained from cards - Sliding window 2 pointer technique
   - Calculate the sum of the first k cards (starting from the left) and assign it to sum
   - Initialize i to k-1 (starting from the last card of the first k cards) and j to n-1 (starting from the last card of the array).
   - Iterate while i and j are within bounds (i >= 0 and j >= 0).
   - Update sum by subtracting the card at index i (moving the left boundary of the window) and adding the card at index j
   -
           int maxScore(vector<int>& arr, int k) {
              int n = arr.size();
              int sum = 0;
              for (int i = 0; i < k; i++) {
                  sum += arr[i];
              }
              int ans = sum;
              int i = k - 1;
              int j = n - 1;
              while (j >= 0 && i >= 0) {
                  sum = sum - arr[i] + arr[j];
                  ans = max(ans, sum);
                  i--;
                  j--;
              }
              return ans;
          }


### 2. Subarray sums equal to k - Hash Table and Prefix Sum
   - Prefix Sum Calculation: Maintain a running total (preSum) as you iterate through the array.
   - Hash Map Usage: Use a hash map to track the frequency of each prefix sum.
   - Count Subarrays: For each element, check if preSum - k exists in the hash map to count subarrays with sum k, and update the hash map with the current preSum
   -
           int subarraySum(vector<int>& nums, int k) {
              map<int,int> mp;
              mp[0]=1;
              int preSum=0;
              int count=0;
              for(int i=0;i<nums.size();i++){
                  preSum+=nums[i];
                  int remove=preSum-k;
                  count+=mp[remove];
                  mp[preSum]++;
              }
              return count;
          }

### 3. Spiral order matrix 
   - set the borders right, left, top, bottom
   - travel directions : left to right, top to bottom, right to left, bottom to  top
   - drecrement the borders eventually
   - 
          vector<int> spiralOrder(vector<vector<int>>& matrix) {
              vector<int>ans;
              if (matrix.empty()) {
                  return ans;
              }
              int dir=0;
              int right=matrix[0].size()-1,left=0,top=0,bottom=matrix.size()-1;
              while(bottom>=top && right>=left)
              {
                  if(dir==0)
                  {
                      for(int i=left;i<=right;i++)
                      {
                      ans.push_back(matrix[top][i]);
                      }
                      dir=1;
                      top++;
                  }
                   else if (dir==1)
                  {
                      for(int i=top;i<=bottom;i++)
                      {
                      ans.push_back(matrix[i][right]);
                      }
                      dir=2;
                      right--;
                  }
                  else if(dir==2)
                  {
                      for(int i=right;i>=left;i--)
                      {
                       ans.push_back(matrix[bottom][i]);
                      }
                      dir=3;
                      bottom--;
                  }
                  else if(dir==3)
                  {
                      for(int i=bottom;i>=top;i--)
                      {
                      ans.push_back(matrix[i][left]);
                      }
                      left++;
                      dir=0;
                  }
              }
              return ans;
          }

### 4. Word Search - Backtracking
   - we can start from any of the position in the matrix. So, we can traverse the complete matrix and choose each of the points as start point
   - if the current size of string that we are forming is equal to the length of the string we need is equal then we just return true
   - if the pointers used i, j get out of bound, or say if we are looking at some point that have already been included (if visited) or if the current position character does not lead us to the answer we return false (WE BACKTRACK)
   -  We make the current position (i, j) visited and then move to the other four directions and if any of those returns true, we'll return true
   -  Make the vis[i][j] = false after all the traversals
   -
              class Solution {
         public:
             int n, m;
             vector<vector<char>> arr;
             string s;
             bool f(int i, int j, int k, vector<vector<bool>> &vis) {
                 if(k == s.length()) return true;
                 if(i < 0 || i >= m || j < 0 || j >= n || arr[i][j] != s[k] || vis[i][j]) return false;
                 vis[i][j] = 1;
                 if(f(i + 1, j, k + 1, vis) || f(i - 1, j, k + 1, vis) || f(i, j + 1, k + 1, vis) || f(i, j - 1, k + 1, vis)) return true;
                 vis[i][j] = 0;
                 return false;
             }
             bool exist(vector<vector<char>>& board, string word) {
                 ios::sync_with_stdio(false);
                 cin.tie(NULL);
                 cout.tie(NULL);
                 arr = board;
                 s = word;
                 m = arr.size();
                 n = arr[0].size();
                 vector<vector<bool>> vis(m, vector<bool>(n, 0));
                 for(int i = 0;i < m;i++) {
                     for(int j = 0;j < n;j++) {
                         if(f(i, j, 0, vis)) return true;
                     }
                 }
                 return false;
             }
         };
      
### 5. Jump Game - Dynammic programming
   - For each index i, first check if i is greater than maxReach. If it is, return false because you cannot move further from this index.
   - Update maxReach to be the maximum of the current maxReach and i + nums[i] (the farthest index that can be reached from i)
   -
               bool canJump(vector<int>& nums) {
                 int n = nums.size();
                 int maxReach = 0;
                 for (int i = 0; i < n; ++i) {
                     if (i > maxReach) {
                         return false;
                     }
                     maxReach = max(maxReach, i + nums[i]);
                 }
                 return maxReach >= n - 1;
             }



## Day - 5 

### 1. All unique permutations of an array - backtracking , recursion
   - Skip if the element is already used.
   - Skip if the element is a duplicate of the previous element and the previous element has not been used.
   - Mark the element as used, add it to the current permutation, and recursively call the backtracking function.
   - After the recursive call, remove the element from the current permutation and mark it as not used (backtrack).
   -
           void backtrack(vector<int> &arr, vector<bool> &used, vector<int> current,vector<vector<int>> &result){
              if(current.size()==arr.size()){
                  result.push_back(current);
                  return;
              }
              for(int i=0;i<arr.size();i++){
                  if(used[i])
                      continue;
                  if(i>0 && arr[i]==arr[i-1] && !used[i-1])
                      continue;
                  used[i]=true;
                  current.push_back(arr[i]);
                  backtrack(arr,used,current,result);
                  current.pop_back();
                  used[i]=false;
              }
          }
          vector<vector<int>> uniquePerms(vector<int> &arr ,int n) {
              sort(arr.begin(),arr.end());
              vector<vector<int>> result;
              vector<int> current;
              vector<bool> used(n,false);
              backtrack(arr,used,current,result);
              return result;
          }


### 2. Game of Live - Stimulation
   -
           void gameOfLife(vector<vector<int>>& board) {
              int n=board.size();
              int m=board[0].size();
              for(int a=0;a<n;a++){
                  for(int b=0;b<m;b++){
                      int count=0;
                      for(int c=a-1;c<=a+1;c++){
                          for(int d=b-1;d<=b+1;d++){
                              if(a==c && b==d) 
                                  continue;
                              if(c>=0 && c<n && d>=0 && d<m && (board[c][d]==1 || board[c][d]==3)){
                                  count++;
                              }
                          }
                      }
                      if(board[a][b]==1 && (count<2 ||count>3)){
                          board[a][b]=3;
                      }
                      if(board[a][b]==0 && count==3){
                          board[a][b]=2;
                      }
                  }
              }
              for(int a=0;a<n;++a){
                  for(int b=0;b<m;++b){
                      if(board[a][b]==3){
                          board[a][b]=0;
                      }
                      if(board[a][b]==2){
                          board[a][b]=1;
                      }
                  }
              }
          }


### 3. Max Value of a equation - Max-heap or priority queue
   - The condition |xi - xj| <= k is maintained by the priority queue to ensure that only valid points are considered.
   - The priority queue helps in keeping track of the maximum (yi - xi) value efficiently.
   - The algorithm ensures that we only consider valid pairs of points and update the result accordingly.
   - 
           int findMaxValueOfEquation(vector<vector<int>>& v, int k) {
              priority_queue<vector<int>> pq;
              pq.push({v[0][1]-v[0][0],v[0][0]});
              int ans = INT_MIN,sum;
              for(int i = 1; i < v.size(); i++){
                  sum = v[i][0]+v[i][1];
                  while(!pq.empty() && v[i][0]-pq.top()[1]>k)
                      pq.pop();
                  if(!pq.empty()){
                      ans = max(ans,sum+pq.top()[0]);
                  }
                  pq.push({v[i][1]-v[i][0],v[i][0]});
              }
              return ans;
          }

### 4. Insert Delete Random O(1) - Hash table
   - bool insert(int val) Inserts an item val into the multiset, even if the item is already present. Returns true if the item is not present, false otherwise.
   - bool remove(int val) Removes an item val from the multiset if present. Returns true if the item is present, false otherwise. Note that if val has multiple occurrences in the multiset, we only remove one of them.
   - int getRandom() Returns a random element from the current multiset of elements.
   -
              class RandomizedCollection {
         public:
             vector<int> nums;
             unordered_map<int,unordered_set<int>> numLocations;
             RandomizedCollection() {
             }
             bool insert(int val) {
                 bool result=true;
                 if(numLocations.contains(val))
                     result=false;
                 nums.push_back(val);
                 numLocations[val].insert(nums.size()-1);
                 return result;    
             }
             bool remove(int val) {
                 if(!numLocations.contains(val))
                     return false;
                 int replaceIndex=*numLocations[val].begin();
                 numLocations[val].erase(replaceIndex);
                 if(numLocations[val].empty())
                     numLocations.erase(val);  
                 int lastValue=nums.back();
                 int lastIndex=nums.size()-1;
                 if(replaceIndex!=lastIndex){
                     swap(nums[replaceIndex],nums[lastIndex]);
                     numLocations[lastValue].erase(lastIndex);
                     numLocations[lastValue].insert(replaceIndex);
                 }
                 nums.pop_back();
                 return true;      
             }
             int getRandom() {
                 return nums[rand()%nums.size()];
             }
         };

### 5. Largest Rectangle in the histogram - Monotonic Stack
   - Each element in the stack is a pair (start, height) where start is the index where the height starts, and height is the height of the bar.
   - Append a zero to heights to ensure that all bars are popped from the stack by the end of the iteration.
   - By maintaining the stack in a way that ensures we only pop when we find a shorter bar, it allows us to calculate the maximum possible area for each height while keeping track of the starting index.
   - 
              int largestRectangleArea(vector<int>& heights) {
                 stack<pair<int,int>> st;
                 heights.push_back(0);
                 int maxArea=0;
                 for(int i=0;i<heights.size();i++){
                     int start=i;
                     while(!st.empty() && st.top().second >heights[i]){
                         auto[idx,height]=st.top();
                         st.pop();
                         maxArea=max(maxArea,height*(i-idx));
                         start=idx;
                     }
                     st.push({start,heights[i]});
                 }
                 return maxArea;
             }
