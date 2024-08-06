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


## Day-6

### 1. Valid Paranthesis - Stack
   - f the current character is an opening bracket (i.e., '(', '{', '['), push it onto the stack.
   -If the current character is a closing bracket (i.e., ')', '}', ']'), check if the stack is empty. If it is empty, return false, because the closing bracket does not have a corresponding opening bracket. Otherwise, pop the top element from the stack and check if it matches the current closing bracket. If it does not match, return false, because the brackets are not valid.

   -
           bool isValid(string s) {
              stack<char> st;
              for(char c:s){
                  if(c=='(' || c=='{' || c=='[')
                  {
                      st.push(c);
                  }else{
                      if(st.empty() ||
                        (c==')' && st.top()!='(') ||
                        (c==']' && st.top()!='[') ||
                        (c=='}' && st.top()!='{')){
                          return false;
                      }
                      st.pop();
                  }
              }
              return st.empty();
          }

### 2. Print all the Duplicate characters in the string - Hashing
   -
           void printDups(string str)
            {
                unordered_map<char, int> count;
                for (int i = 0; i < str.length(); i++) {
                    count[str[i]]++;
                }
                for (auto it : count) {
                    if (it.second > 1)
                        cout << it.first << ", count = " << it.second
                             << "\n";
                }
            }

### 3. Find first index of needle in the haystack - two pointers
   - If the current characters in haystack and needle match (haystack[fir] == needle[sec])
   - Check if sec is at the last character of needle (sec == nNeedle - 1). If true, return the starting index of the match (fir - sec)
   - If characters do not match after the comparison (either directly or after an increment), adjust the fir pointer. Move fir back to the next character after the start of the current match attempt (fir = (fir - sec) + 1). Reset sec to 0 to start matching needle from the beginning
   -
           int strStr(string haystack, string needle) {
              int fir=0,sec=0;
              int nstack=haystack.size(),nNeedle=needle.size();
              while(fir<nstack){
                  if(haystack[fir]==needle[sec]){
                      if(sec==nNeedle-1){
                          return fir-sec;
                      }
                      sec++;
                  }
                  fir++;
                  if(haystack[fir]!=needle[sec]){
                      fir=(fir-sec)+1;
                      sec=0;
                  }
              }
              return -1;
          }
          

### 4. Check if a string is palindrome with maximum one replacement possible - two pointers and greedy
   - if its the first time they didn't match, delete any one of them left and right and see if it comes palindrome
   -
           bool palin(string s,int i,int j){
              while(j>=i){
                  if(s[i]!=s[j]){
                      return false;
                  }else{
                      i++;
                      j--;
                  }
              }
              return true;
          }
          bool validPalindrome(string s) {
              int i=0;
              int j=s.length()-1;
              while(j>=i){
                  if(s[i]!=s[j]){
                      return palin(s,i+1,j) || palin(s,i,j-1);
                  }else{
                      i++;
                      j--;
                  }
              }
              return true;
          }

### 5. Integer to Roman - Hash Table
   -
           string intToRoman(int num) {
              string ones[] = {"","I","II","III","IV","V","VI","VII","VIII","IX"};
              string tens[] = {"","X","XX","XXX","XL","L","LX","LXX","LXXX","XC"};
              string hrns[] = {"","C","CC","CCC","CD","D","DC","DCC","DCCC","CM"};
              string ths[]={"","M","MM","MMM"};
              return ths[num/1000]+hrns[(num%1000)/100]+tens[(num%100)/10]+ones[num%10];
          }

### 6. Generate Paranthesis - Recursion
   -
           void solve(int total,int open,int close,string s,vector<string> &ans){
              if(s.size()==total){
                  ans.push_back(s);
                  return;
              }
              if(open>close){
                  solve(total,open,close+1,s+')',ans);
                  if(open<total/2){
                      solve(total,open+1,close,s+'(',ans);
                  }
              }else{
                  solve(total,open+1,close,s+'(',ans);
              }
          }
          vector<string> generateParenthesis(int n) {
              vector<string> ans;
              solve(n*2,0,0,"",ans);
              return ans;
          }      
   


## Day-7

### 1. Simplify Path - Stack
   - Whenever we encounter any file’s name, we simply push it into the stack.
   - when we come across ” . ” we do nothing
   - When we find “..” in our path, we simply pop the topmost element as we have to jump back to parent’s directory
   -
           string simplifyPath(string path) {
              stack<string> st;
              string res;
              for(int i=0;i<path.size();i++){
                  if(path[i]=='/')
                      continue;
                  string temp;
                  while(i<path.size() && path[i]!='/'){
                      temp+=path[i];
                      ++i;
                  }
                  if(temp==".")
                      continue;
                  else if(temp==".."){
                      if(!st.empty())
                          st.pop();
                  }else{
                      st.push(temp);
                  }        
              }
              while(!st.empty()){
                  res="/"+st.top()+res;
                  st.pop();
              }
              if(res.size()==0)
                  return "/";
              else
                  return res;    
          }

### 2. Smallest window in a string containing all the characters of another string - Stack
   - Expand the Window: Move the right pointer to expand the window.Decrease counter if the current character is in t.Decrease the count of the current character in the map.
   - Shrink the Window:If counter is 0 (valid window), move the left pointer to shrink the window.Update minStart and minLength if a smaller window is found.Increase the count of the current character in the map and counter if the character is in t.
   -
           string minWindow(string s, string t) {
              unordered_map<char, int> m;
              for (int i = 0; i < t.size(); i++) {
                  m[t[i]]++;
              }
              int left = 0;
              int right = 0;
              int counter = t.size();
              int minStart = 0;
              int minLength = INT_MAX;
              while (right < s.size()) {
                  if (m[s[right]] > 0) {
                      counter--;
                  }
                  m[s[right]]--;
                  right++;
                  while (counter == 0) {
                      if (right - left < minLength) {
                          minStart = left;
                          minLength = right - left;
                      }
                      m[s[left]]++;
                      if (m[s[left]] > 0) {
                          counter++;
                      }
                      left++;
                  }
              }
              if (minLength != INT_MAX) {
                  return s.substr(minStart, minLength);
              }
              return "";
          }

### 3. Reverse words in a String - two pointers
   - Identify Word Boundaries: Traverse the input string to identify the start and end positions of each word. Skip over any spaces to locate the beginning of a word. Move through the characters until a space is encountered to mark the end of the word. Store the start and end indices of each word in a vector.
   - Reverse the Order of Words: Iterate over the stored word positions in reverse order to reconstruct the final string. For each word, extract the substring using the stored indices and append it to the result string. Ensure that words are concatenated with a single space in between.
   - Return the Result: The final string, built by appending the words in reverse order, is returned as the output.
   -
           vector<pair<int,int>> wordPositions;
              int i=0;
              while(i<length){
                  while(i<length && s[i]==' ')
                      i++;
                  if(i==length)
                      break;
                  int start=i;
                  while(i<length && s[i]!=' ')
                      i++;
                  int end=i-1;
                  wordPositions.push_back({start,end});    
              }
              string result="";
              for(int j=wordPositions.size()-1;j>=0;j--){
                  string word=s.substr(wordPositions[j].first,wordPositions[j].second -wordPositions[j].first+1);
                  result+=word;
                  if(j!=0) 
                      result+=" ";
              }
              return result;
          } 


### 4. Pattern Search - Rabin Karp Algorithm
   - Rabin Karp algorithm matches the hash value of the pattern with the hash value of the current substring of text, and if the hash values match then only it starts matching individual characters.
   - The hash value is calculated using a rolling hash function, which allows you to update the hash value for a new substring by efficiently removing the contribution of the old character and adding the contribution of the new character. This makes it possible to slide the pattern over the text and calculate the hash value for each substring without recalculating the entire hash from scratch.
   - Choose a suitable base and a modulus: Select a prime number ‘p‘ as the modulus. This choice helps avoid overflow issues and ensures a good distribution of hash values. Choose a base ‘b‘ (usually a prime number as well), which is often the size of the character set (e.g., 256 for ASCII characters)
   - each character ‘c’ at position ‘i’, calculate its contribution to the hash value as ‘c * (bpattern_length – i – 1) % p’ and add it to ‘hash‘. This gives you the hash value for the entire pattern.
   - Slide the pattern over the text: Start by calculating the hash value for the first substring of the text that is the same length as the pattern
   -
            void search(char pat[], char txt[], int q)
            {
                int M = strlen(pat);
                int N = strlen(txt);
                int i, j;
                int p = 0;
                int t = 0;
                int h = 1;
                // The value of h would be "pow(d, M-1)%q"
                for (i = 0; i < M - 1; i++)
                    h = (h * d) % q;
                for (i = 0; i < M; i++) {
                    p = (d * p + pat[i]) % q;
                    t = (d * t + txt[i]) % q;
                }
                for (i = 0; i <= N - M; i++) {
                    if (p == t) {
                        for (j = 0; j < M; j++) {
                            if (txt[i + j] != pat[j]) {
                                break;
                            }
                        }
                        if (j == M)
                            cout << "Pattern found at index " << i
                                 << endl;
                    }
                    if (i < N - M) {
                        t = (d * (t - txt[i] * h) + txt[i + M]) % q;
                        if (t < 0)
                            t = (t + q);
                    }
                }
            }


## Day-8

### 1. Group Anagrams - Hash Table
   - After sorting a group of anagrams, their sorted words will be the same
   - So we can group the anagrams using their sorted words as the key of a hash map
   -
           vector<vector<string>> groupAnagrams(vector<string>& strs) {
              unordered_map<string,vector<string>> mp;
              for(auto s:strs){
                  string word=s;
                  sort(word.begin(),word.end());
                  mp[word].push_back(s);
              }
              vector<vector<string>> ans;
              for(auto x:mp){
                  ans.push_back(x.second);
              }
              return ans;
          }

### 2. Word Wrap - Dynammic programming
   -   Declare variables n as the length of the nums array, and i, j for iteration.
       Initialize variables currlen to store the number of characters in the current line, and cost to store the possible minimum cost of the line.
       Create dynamic programming (DP) table dp of size n to represent the cost of lines starting with each word.
       Create an array ans to store the index of the last word in each line.
       Base Case:
           Set dp[n - 1] to 0 since the last line has no extra cost, and ans[n - 1] to n - 1 as it is the last word.
       Iterate from i = n - 2 to 0 (backwards) to fill the DP table.
       For each word at index i, try adding words from j = i to n-1 to the current line.
       Calculate the currlen by adding the length of words and spaces.
       If the limit k is violated, break the loop.
       Calculate the cost of the line and update dp[i] and ans[i] if the current arrangement minimizes the cost.
       Initialize res to 0 and iterate through the ans array to calculate the total cost.
       For each line, sum the lengths of the words and calculate the extra spaces (pos).
       Add the square of extra spaces to the total cost.
   -
          int solveWordWrap(vector<int>nums, int k) 
          { 
              int n = nums.size();
              int i, j;
              int currlen;
              int cost;
              vector<int>dp(n, 0);
              vector<int>ans(n, 0);
              dp[n - 1] = 0;
              ans[n - 1] = n - 1;
              for (i = n - 2; i >= 0; i--)
              {
                  currlen = -1;
                  dp[i] = INT_MAX;
                  for (j = i; j < n; j++)
                  {
                      currlen += (nums[j] + 1);
                      if (currlen > k)
                          break;
                      if (j == n - 1)
                          cost = 0;
                      else
                          cost = (k - currlen) * (k - currlen) + dp[j + 1];
                      if (cost < dp[i]) {
                          dp[i] = cost;
                          ans[i] = j;
                      }
                  }
              }
              int res = 0;
              i = 0;
              while (i < n) {
                  int pos = 0;
                  for (int j = i; j < ans[i] + 1; j++) {
                      pos += nums[j];
                  }
                  int x = ans[i]-i;
                  if(ans[i]+1 != nums.size())
                      res += (k - x - pos)*(k - x - pos);
                  i = ans[i] + 1;
              }
              return res;
          } 


### 3. Basic Calculator - Stack
   - To begin with, we add '+' to the s=input string
   - if we encounter a digit/num (The string might also contain more than 1 digit numbers, so until we find any operator, we'll keep on forming the multi digit number (num = num*10 + s[i] ) ).
   - At the end, we encounter another '+' sign, this is to push the last curr value into the stack
   -
              int calculate(string s) {
                 s+='+';
                 stack<int> st;
                 long long int ans=0,curr=0;
                 char sign='+';
                 for(int i=0;i<s.size();i++){
                     if(isdigit(s[i]))
                         curr=curr*10+(s[i]-'0');
                     else if(s[i]=='+' || s[i]=='-' || s[i]=='*' || s[i]=='/'){
                         if(sign=='+')
                             st.push(curr);
                         else if(sign=='-')
                             st.push(-1*curr);
                         else if(sign=='*'){
                             int num=st.top();st.pop();
                             st.push(curr*num);
                         }else if(sign=='/'){
                             int num=st.top();st.pop();
                             st.push(num/curr);
                         }
                         curr=0;
                         sign=s[i];        
                     }
                 }
                 while(st.size()){
                     ans+=st.top();
                     st.pop();
                 }
                 return ans;
             }

### 4. Valid Number - String Finite State Machine
   -  Initializes the state to START.
      Iterates through each character of the input string s.
      Uses the parse function to get the Token for the current character.
      Uses the transition table to determine the next state based on the current state and Token.
      If the state transitions to FAIL, the function returns false indicating the string is not a valid number.
      After processing all characters, returns true if the final state is a valid accepting state (a state less than START).
   -
              class Solution {
         public:
             enum State {
                 EXPONENT,
                 DECIMAL,
                 INTEGER,
                 START,
                 EMPTY_INTEGER,
                 EMPTY_DECIMAL,
                 EXPONENT_START,
                 EMPTY_EXPONENT,
                 FAIL,
             };
             enum Token {
                 SIGN,
                 DIGIT,
                 DOT,
                 EXP,
                 OTHER,
             };
             State transition[FAIL][OTHER] = {
                 /* EXPONENT */       {FAIL, EXPONENT, FAIL, FAIL},
                 /* DECIMAL */        {FAIL, DECIMAL, FAIL, EXPONENT_START},
                 /* INTEGER */        {FAIL, INTEGER, DECIMAL, EXPONENT_START},
                 /* START */          {EMPTY_INTEGER, INTEGER, EMPTY_DECIMAL, FAIL},
                 /* EMPTY_INTEGER */  {FAIL, INTEGER, EMPTY_DECIMAL, FAIL},
                 /* EMPTY_DECIMAL */  {FAIL, DECIMAL, FAIL, FAIL},
                 /* EXPONENT_START */ {EMPTY_EXPONENT, EXPONENT, FAIL, FAIL},
                 /* EMPTY_EXPONENT */ {FAIL, EXPONENT, FAIL, FAIL},
             };
             Token parse(char c) {
                 switch (c) {
                     case '+':
                     case '-': return SIGN;
                     case '.': return DOT;
                     case '0':
                     case '1':
                     case '2':
                     case '3':
                     case '4':
                     case '5':
                     case '6':
                     case '7':
                     case '8':
                     case '9': return DIGIT;
                     case 'e':
                     case 'E': return EXP;
                     default: return OTHER;
                 }
             }
             bool isNumber(string s) {
                 State state=START;
                 for(const char c:s){
                     const Token ch=parse(c);
                     if(ch==OTHER){
                         return false;
                     }
                     state=transition[state][ch];
                     if(state==FAIL){
                         return false;
                     }
                 }
                 return (state<START);
             }
         };

   -
      bool num = false, exp = false, sign = false, dec = false;
           for (auto c : S)
               if (c >= '0' && c <= '9') num = true ;    
               else if (c == 'e' || c == 'E')
                   if (exp || !num) return false;
                   else exp = true, sign = false, num = false, dec = false;
               else if (c == '+' || c == '-')
                   if (sign || num || dec) return false;
                   else sign = true;
               else if (c == '.')
                   if (dec || exp) return false;
                   else dec = true;
               else return false;
           return num;


### 5. Integer to English words - Recursion
   - Calculate the Quotient: If value is 100 or more, the function recursively calls numberToWords(num / value) to convert the quotient to words. This handles the part before the current value (e.g., the "One" in "One Thousand").
   - Append the Current Word: The current word corresponding to value is appended to the result.
   - Calculate the Remainder: If there's a remainder (i.e., num % value is not zero), the function recursively calls numberToWords(num % value) to convert the remainder to words and appends it to the result.
   -
              vector<pair<int, string>> mp = {
                 {1000000000, "Billion"},{1000000, "Million"},{1000, "Thousand"},{100, "Hundred"},
                 {90, "Ninety"},{80, "Eighty"},{70, "Seventy"},{60, "Sixty"},{50, "Fifty"},
                 {40, "Forty"},{30, "Thirty"},{20, "Twenty"},{19, "Nineteen"},{18, "Eighteen"},
                 {17, "Seventeen"},{16, "Sixteen"},{15, "Fifteen"},{14, "Fourteen"},
                 {13, "Thirteen"},{12, "Twelve"},{11, "Eleven"},{10, "Ten"},{9, "Nine"},
                 {8, "Eight"},{7, "Seven"},{6, "Six"},{5, "Five"},{4, "Four"},{3, "Three"},
                 {2, "Two"},{1, "One"}
             };
             string numberToWords(int num) {
                 if(num==0)
                     return "Zero";
                 for(auto i:mp){
                     if(num>=i.first){
                         string a="";
                         if(num>=100){
                             a=numberToWords(num/i.first)+" ";
                         }
                         string b=i.second;
                         string c="";
                         if(num%i.first!=0){
                             c=" "+numberToWords(num%i.first);
                         }
                         return a+b+c;
                     }
                 }
                 return "";    
             }



## Day-9

### 1. Text Justification - String stimulation
   - Justify the current line:
    Calculate totalSpaces and spacesBetweenWords.
    If spacesBetweenWords is 0, left-justify the single word.
    Distribute spaces evenly among words.
    Append justified line to result.

   - Add the word to the current line:
    Add the current word to currentLine.
    Update currentLineLength.

   - Left-justify the last line:
    Concatenate words in currentLine with a single space between them.
    Pad the end with spaces to reach maxWidth.
    Add the last line to result
   -
           vector<string> fullJustify(vector<string>& words, int maxWidth) {
           vector<string> result;
           vector<string> currentLine;
           int currentLineLength = 0;
           for (const string& word : words) {
               if (currentLineLength + word.length() + currentLine.size() > maxWidth) {
                   int totalSpaces = maxWidth - currentLineLength;
                   int spacesBetweenWords = currentLine.size() - 1;
                   if (spacesBetweenWords == 0) {
                       result.push_back(currentLine[0] + string(totalSpaces, ' '));
                   } else {
                       int spacesPerSlot = totalSpaces / spacesBetweenWords;
                       int extraSpaces = totalSpaces % spacesBetweenWords;
                       string line;
                       for (int i = 0; i < currentLine.size(); ++i) {
                           line += currentLine[i];
                           if (i < spacesBetweenWords) {
                               line += string(spacesPerSlot + (i < extraSpaces ? 1 : 0), ' ');
                           }
                       }
                       result.push_back(line);
                   }
                   currentLine.clear();
                   currentLineLength = 0;
               }
               currentLine.push_back(word);
               currentLineLength += word.length();
           }
           string lastLine;
           for (int i = 0; i < currentLine.size(); ++i) {
               lastLine += currentLine[i];
               if (i < currentLine.size() - 1) {
                   lastLine += " ";
               }
           }
           lastLine += string(maxWidth - lastLine.length(), ' ');
           result.push_back(lastLine);
           return result;
       }

### 2. Boyer Moore Algorithm for Pattern Searching
   - Mismatch become match : We will lookup the position of the last occurrence of the mismatched character in the pattern, and if the mismatched character exists in the pattern, then we’ll shift the pattern such that it becomes aligned to the mismatched character in the text T.
   - 
            #include <bits/stdc++.h>
            using namespace std;
            #define NO_OF_CHARS 256
            // The preprocessing function for Boyer Moore's bad character heuristic
            void badCharHeuristic(string str, int size,int badchar[NO_OF_CHARS])
            {
            	int i;
            	// Initialize all occurrences as -1
            	for (i = 0; i < NO_OF_CHARS; i++)
            		badchar[i] = -1;
            	// Fill the actual value of last occurrence
            	// of a character
            	for (i = 0; i < size; i++)
            		badchar[(int)str[i]] = i;
            }
            /* A pattern searching function that uses BadCharacter Heuristic of Boyer Moore Algorithm */
            void search(string txt, string pat)
            {
            	int m = pat.size();
            	int n = txt.size();
            	int badchar[NO_OF_CHARS];
            	/* Fill the bad character array by calling the preprocessing function badCharHeuristic() for given pattern */
            	badCharHeuristic(pat, m, badchar);
            	int s = 0; // s is shift of the pattern withrespect to text
            	while (s <= (n - m)) {
            		int j = m - 1;
            		/* Keep reducing index j of pattern whilecharacters of pattern and text arematching at this shift s */
            		while (j >= 0 && pat[j] == txt[s + j])
            			j--;
            		/* If the pattern is present at currentshift, then index j will become -1 afterthe above loop */
            		if (j < 0) {
            			cout << "pattern occurs at shift = " << s<< endl;
            			/* Shift the pattern so that the nextcharacter in text aligns with the lastoccurrence of it in pattern.The condition s+m < n is necessary forthe case when pattern occurs at the endof text */
            			s += (s + m < n) ? m - badchar[txt[s + m]] : 1;
            		}
            		else
            			/* Shift the pattern so that the bad characterin text aligns with the last occurrence ofit in pattern. The max function is used to make sure that we get a positive shift. We may get a negative shift if the last occurrence of bad character in pattern is on the right side of the current character. */
            			s += max(1, j - badchar[txt[s + j]]);
            	}
            }

### 3. Distinct sequences - Dynammic programming
   - dp is a 2D vector (table) of size (m+1) x (n+1) initialized to 0. This table will be used to store the number of distinct subsequences of substrings of s that match substrings of t
   - If t is an empty string, there is exactly one subsequence of any substring of s that matches it: the empty subsequence. Thus, dp[i][0] = 1 for all i.
   - If s is an empty string and t is non-empty, there are no subsequences of s that match t. Thus, dp[0][j] = 0 for all j > 0
   - We use this character in s to match the character in t, so we add the number of ways to match the previous characters (dp[i - 1][j - 1]).
   - We do not use this character in s, so we add the number of ways to match t with the previous characters in s (dp[i - 1][j])
   -
          int numDistinct(string s, string t) {
              int m=s.size(),n=t.size();
              vector<vector<double>> dp(m+1,vector<double>(n+1,0));
              for(int i=0;i<=m;i++)
                  dp[i][0]=1;
              for(int j=1;j<=n;j++)
                  dp[0][j]=0;
              for(int i=1;i<=m;i++){
                  for(int j=1;j<=n;j++){
                      if(s[i-1]==t[j-1]){
                          dp[i][j]=dp[i-1][j-1]+dp[i-1][j];
                      }else{
                          dp[i][j]=dp[i-1][j];
                      }
                  }
              }
              return (int) dp[m][n];        
          }

### 4. Print Anagrams together - Hash
   -
           vector<vector<string> > Anagrams(vector<string>& string_list) {
              vector<vector<string>> result;
              unordered_map<string,vector<string>> mp;
              for(string s:string_list){
                  string s2=s;
                  sort(s.begin(),s.end());
                  mp[s].push_back(s2);
              }
              for( auto itr=mp.begin();itr!=mp.end();itr++){
                  result.push_back(itr->second);
              }
              return result;
          }


## Day-10
### Aptitude preparation
  - Logical reasoning
  - Patterns
  - Data interpretation
  - C++

    

## Day-11

### 1. Minimum moves to equal array elements - Array Sorting
   -
              int minMoves2(vector<int>& nums) {
                 sort(nums.begin(),nums.end());
                 int n=nums.size(),mid=n/2,ans=0;
                 int dif=nums[mid];
                 for(int i=0;i<n;i++){
                     ans+=abs(dif-nums[i]);
                 }
                 return ans;
             }

### 2. Add binary - Bit manupilation
   -
              string addBinary(string a, string b) {
                 string bit,ans;
                 int carry=0,i=a.length()-1,j=b.length()-1;
                 while(i>=0 || j>=0 || carry){
                     if(i>=0)
                         carry+=a[i--]-'0';
                     if(j>=0)
                         carry+=b[j--]-'0';
                     bit=carry%2+'0';
                     ans=bit+ans;
                     carry/=2;    
                 }
                 return ans;
             }

### 3. Maximum product of three numbers - Array and negatives case handling
   -
              int maximumProduct(vector<int>& nums) {
                 int max1=INT_MIN,max2=INT_MIN,max3=INT_MIN;
                 int min1=INT_MAX,min2=INT_MAX;
                 for(int num:nums){
                     if(num>max1){
                         max3=max2;
                         max2=max1;
                         max1=num;
                     }else if(num>max2){
                         max3=max2;
                         max2=num;
                     }else if(num>max3){
                         max3=num;
                     }
                     if(num<min1){
                         min2=min1;
                         min1=num;
                     }else if(num<min2){
                         min2=num;
                     }
                 }
                 return max(max1*max2*max3,min1*min2*max1);
             }

### 4. Excel sheet column title - String Math
   -
              string convertToTitle(int columnNumber) {
                 string str="";
                 while(columnNumber>0){
                     char ch=char((columnNumber-1)%26+65);
                     str=ch+str;
                     columnNumber=(columnNumber-1)/26;
                 }
                 return str;
             }

### 5. Product Array Puzzle - Math
   -
              vector<long long int> productExceptSelf(vector<long long int>& nums, int n) {
                vector<long long int> result(n, 1);
                 long long int leftProduct = 1;
                 for (int i = 0; i < n; ++i) {
                     result[i] = leftProduct;
                     leftProduct *= nums[i];
                 }
                 long long int rightProduct = 1;
                 for (int i = n - 1; i >= 0; --i) {
                     result[i] *= rightProduct;
                     rightProduct *= nums[i];
                 }
                 return result;
             }

### 6. Maximum size rectangle binary sub-matrix with all 1s - Matrix Histogram
   - If the height of bars of the histogram is given then the largest area of the histogram can be found. This way in each row, the largest area of bars of the histogram can be found. To get the largest rectangle full of 1’s, update the next row with the previous row and find the largest area under the histogram, i.e. consider each 1’s as filled squares and 0’s with an empty square and consider each row as the base.
   - Run a loop to traverse through the rows.
   - Now If the current row is not the first row then update the row as follows, if matrix[i][j] is not zero then matrix[i][j] = matrix[i-1][j] + matrix[i][j].
   - https://www.geeksforgeeks.org/maximum-size-rectangle-binary-sub-matrix-1s/
   -
           int maxRectangle(int A[][C])
         {
             // Calculate area for first row and initialize it as
             // result
             int result = maxHist(A[0]);
             // iterate over row to find maximum rectangular area
             // considering each row as histogram
             for (int i = 1; i < R; i++) {
                 for (int j = 0; j < C; j++)
                     // if A[i][j] is 1 then add A[i -1][j]
                     if (A[i][j])
                         A[i][j] += A[i - 1][j];
                 // Update result if area with current row (as last
                 // row) of rectangle) is more
                 result = max(result, maxHist(A[i]));
             }
             return result;
         }

### 7. Number of Islands using DFS - Graphs
   - The idea is to keep an additional matrix to keep track of the visited nodes in the given matrix, and perform DFS to find the total number of islands
   -
              void DFS(vector<vector<int> >& M, int i, int j, int ROW,
                  int COL)
               {
                   // Base condition
                   // if i less than 0 or j less than 0 or i greater than
                   // ROW-1 or j greater than COL-  or if M[i][j] != 1 then
                   // we will simply return
                   if (i < 0 || j < 0 || i > (ROW - 1) || j > (COL - 1)
                       || M[i][j] != 1) {
                       return;
                   }
                   if (M[i][j] == 1) {
                       M[i][j] = 0;
                       DFS(M, i + 1, j, ROW, COL); // right side traversal
                       DFS(M, i - 1, j, ROW, COL); // left side traversal
                       DFS(M, i, j + 1, ROW, COL); // upward side traversal
                       DFS(M, i, j - 1, ROW,
                           COL); // downward side traversal
                       DFS(M, i + 1, j + 1, ROW,
                           COL); // upward-right side traversal
                       DFS(M, i - 1, j - 1, ROW,
                           COL); // downward-left side traversal
                       DFS(M, i + 1, j - 1, ROW,
                           COL); // downward-right side traversal
                       DFS(M, i - 1, j + 1, ROW,
                           COL); // upward-left side traversal
                   }
               }
               int countIslands(vector<vector<int> >& M)
               {
                   int ROW = M.size();
                   int COL = M[0].size();
                   int count = 0;
                   for (int i = 0; i < ROW; i++) {
                       for (int j = 0; j < COL; j++) {
                           if (M[i][j] == 1) {
                               count++;
                               DFS(M, i, j, ROW, COL); // traversal starts
                                                       // from current cell
                           }
                       }
                   }
                   return count;
               }

### 8. Given a matrix of ‘O’ and ‘X’, replace ‘O’ with ‘X’ if surrounded by ‘X’ - Matrix
   -
                 void floodFillUtil(char mat[][N], int x, int y, char prevV, char newV)
                  {
                      // Base cases
                      if (x < 0 || x >= M || y < 0 || y >= N)
                          return;
                      if (mat[x][y] != prevV)
                          return;
                      // Replace the color at (x, y)
                      mat[x][y] = newV;
                      // Recur for north, east, south and west
                      floodFillUtil(mat, x+1, y, prevV, newV);
                      floodFillUtil(mat, x-1, y, prevV, newV);
                      floodFillUtil(mat, x, y+1, prevV, newV);
                      floodFillUtil(mat, x, y-1, prevV, newV);
                  }
                  // Returns size of maximum size subsquare matrix surrounded by 'X'
                  int replaceSurrounded(char mat[][N])
                  {
                     // Step 1: Replace all 'O'  with '-'
                     for (int i=0; i<M; i++)
                        for (int j=0; j<N; j++)
                            if (mat[i][j] == 'O')
                               mat[i][j] = '-';
                     // Call floodFill for all '-' lying on edges
                     for (int i=0; i<M; i++)   // Left side
                        if (mat[i][0] == '-')
                          floodFillUtil(mat, i, 0, '-', 'O');
                     for (int i=0; i<M; i++)  //  Right side
                        if (mat[i][N-1] == '-')
                          floodFillUtil(mat, i, N-1, '-', 'O');
                     for (int i=0; i<N; i++)   // Top side
                        if (mat[0][i] == '-')
                          floodFillUtil(mat, 0, i, '-', 'O');
                     for (int i=0; i<N; i++)  // Bottom side
                        if (mat[M-1][i] == '-')
                          floodFillUtil(mat, M-1, i, '-', 'O');
                     // Step 3: Replace all '-' with 'X'
                     for (int i=0; i<M; i++)
                        for (int j=0; j<N; j++)
                           if (mat[i][j] == '-')
                               mat[i][j] = 'X';
                  }

### 9. Rotate a matrix - Matrix
   - 2 Steps to rotate image
   - Transpose the matrix
   - Swap the columns
   -
           void rotate(vector<vector<int>>& matrix) {
              int row=matrix.size();
              for(int i=0;i<row;i++){
                  for(int j=0;j<=i;j++){
                      swap(matrix[i][j],matrix[j][i]);
                  }
              }
              for(int i=0;i<row;i++){
                  reverse(matrix[i].begin(),matrix[i].end());
              }
          }
   - 
           void rotate(vector<vector<int>>& matrix) {
              int n=matrix.size();
              for(int row=0;row<n/2;row++){
                  for(int col=row;col<n-row-1;col++){
                      swap(matrix[row][col],matrix[col][n-1-row]);
                      swap(matrix[row][col],matrix[n-1-row][n-1-col]);
                      swap(matrix[row][col],matrix[n-1-col][row]);
                  }
              }
          }


## Day - 12

### 1. Permute two arrays such that sum of every pair is greater or equal to K - Sorting
   - The idea is to sort one array in ascending order and another array in descending order and if any index does not satisfy the condition a[i] + b[i] >= K then print “No”, else print “Yes”.
   -
           bool isPossible(int a[], int b[], int n, int k)
            {
                sort(a, a + n);
                sort(b, b + n, greater<int>());
                for (int i = 0; i < n; i++)
                    if (a[i] + b[i] < k)
                        return false;
                return true;
            }

### 2. Max number of K sum pairs - Two pointers
   - Sort the input array nums to bring similar elements together.
   - Initialize two pointers, i at the beginning (0) and j at the end (n-1) of the array.
   - Iterate until i is less than j:
    If nums[i] + nums[j] equals k, increment cnt and move i and j pointers towards each other.
    If nums[i] + nums[j] is greater than k, decrement j to reduce the sum.
    If nums[i] + nums[j] is less than k, increment i to increase the sum.
   -
              int maxOperations(vector<int>& nums, int k) {
                 int n=nums.size();
                 sort(nums.begin(),nums.end());
                 int i=0,j=n-1,cnt=0;
                 while(i<j){
                     if((nums[i]+nums[j])==k) {
                         cnt++;
                         i++;
                         j--;
                     }
                     else if((nums[i]+nums[j]) >k){
                         j--;
                     }
                     else i++;
                 }
                 return cnt;
             }

### 3. Find pair given difference - Two pointers
   - Two pointers, i and j, are initialized at indices 0 and 1 respectively.
   - A while loop is used to iterate until one of the pointers goes out of bounds.
   - Inside the loop:
    If the difference between the elements at i and j is equal to n, return true.
    If the difference is less than n, increment j.
    If the difference is greater than n, increment i.
   -
           int findPair(int n, int x, vector<int> &arr) {
              sort(arr.begin(), arr.end());
              int i = 0;
              int j = 1;
              while (i < n && j < n) {
                  if (i != j && arr[j] - arr[i] == x) {
                      return 1;
                  } else if (arr[j] - arr[i] < x) {
                      j++;
                  } else {
                      i++;
                  }
              }
              return -1;
          }


### 4. Ceiling in a sorted array - Binary search
   - break once condition break loop will return start and ans is low which will be next smallest greater than target which is ceiling
   - 
           int ceilSearch(int arr[], int low, int high, int x)
            {
                if (arr.size() == 0) {
                    return -1;
                }
                int mid;
                while (low <= high) {
                    mid = low + (high - low) / 2;
                    if (arr[mid] == x)
                        return mid;
                    else if (x < arr[mid])
                        high = mid - 1;
                    else
                        low = mid + 1;
                }
                return low;
            }

### 5. Check if reversing a sub array make the array sorted - Sorting
   - Initialize two variables x and y with -1.
   - Iterate over the array.
       Find the first number for which a[i] > a[i+1] and store it into x. 
       Similarly, Store index i+1 as well into y, As this will keep track of the ending of the subarray which is needed to reverse.
   - Check if x == -1 then array is already sorted so return true.
       Otherwise, reverse the array from index x to index y.
       Traverse the array to check for every element is sorted or not
   -
           bool sortArr(int a[], int n) 
            { 
                int x = -1; 
                int y = -1; 
                for (int i = 0; i < n - 1; i++) { 
                    if (a[i] > a[i + 1]) { 
                        if (x == -1) { 
                            x = i; 
                        } 
                        y = i + 1; 
                    } 
                } 
                if (x != -1) { 
                    reverse(a + x, a + y + 1); 
                    for (int i = 0; i < n - 1; i++) { 
                        if (a[i] > a[i + 1]) { 
                            return false; 
                            return 0; 
                        } 
                    } 
                } 
                return true; 
            } 

### 6. Radix Sort - Sorting
   - Radix Sort is a linear sorting algorithm that sorts elements by processing them digit by digit. It is an efficient sorting algorithm for integers or strings with fixed-size keys.
   - It assumes that sorting numbers digit by digit will eventually result in a fully sorted list.
   -
           void countSort(int arr[], int n, int exp)
            {
                // Output array int output[n];
                int i, count[10] = { 0 };
                // Store count of occurrences in count[]
                for (i = 0; i < n; i++)
                    count[(arr[i] / exp) % 10]++;
                // Change count[i] so that count[i] now contains actual position of this digit in output[]
                for (i = 1; i < 10; i++)
                    count[i] += count[i - 1];
                // Build the output array
                for (i = n - 1; i >= 0; i--) {
                    output[count[(arr[i] / exp) % 10] - 1] = arr[i];
                    count[(arr[i] / exp) % 10]--;
                }
                for (i = 0; i < n; i++)
                    arr[i] = output[i];
            }
            void radixsort(int arr[], int n)
            {
                int m = getMax(arr, n);
                for (int exp = 1; m / exp > 0; exp *= 10)
                    countSort(arr, n, exp);
            }


## Day - 13

### 1. Find Peak Element - Binary Search and Sorting
   - To reduce the search space and make it more efficient we can do the following things:-
(i): check if there is only one element in the array, if it is, then return the first index(0) itself.
(ii): If the first element is greater than the second element then again return the first index(0). (reason -> read the 3rd point in problem description).
(iii): If the last element is greater than second last element, then return the last index(n-1).
   - Now simply apply Binary Search taking variables start = 0, end = n-2(because we already checked the last element so n-2), and mid.
   - If mid element is greater than mid+1, then shift to left side of array reducing the sarch space. Else start from mid+1 the same procedure in a loop.
   -
              int findPeakElement(vector<int>& nums) {
                 int n=nums.size();
                 if(n==1 || nums[0]>nums[1])
                     return 0;
                 if(nums[n-1]>nums[n-2])
                     return n-1;
                 int start=0,end=n-2;
                 while(start<end){
                     int mid=start+(end-start)/2;
                     if(nums[mid]>nums[mid+1]){
                         end=mid;
                     }else{
                         start=mid+1;
                     }
                 }  
                 return start;  
             }


### 2. Allocate Minimum Pages - Binary search
   - When no valid answer exists: If the number of students is greater than the number of books (i.e, M > N), In this case at least 1 student will be left to which no book has been assigned.
   - When a valid answer exists.
    The maximum possible answer could be when there is only one student. So, all the book will be assigned to him and the result would be the sum of pages of all the books.
    The minimum possible answer could be when number of student is equal to the number of book (i.e, M == N) , In this case all the students will get at most one book. So, the result would be the maximum number of pages among them (i.e, maximum(pages[])).
   - Hence, we can apply binary search in this given range and each time we can consider the mid value as the maximum limit of pages one can get. And check for the limit if answer is valid then update the limit accordingly.
   -
              bool isPossible(int arr[],int n,int m,long long curMin){
                 int studentReq=1;
                 long long curSum=0;
                 for(int i=0;i<n;i++){
                     if(arr[i]>curMin)
                         return false;
                     if(curSum+arr[i]>curMin){
                         studentReq++;
                         curSum=arr[i];
                         if(studentReq>m)
                             return false;
                     }else{
                         curSum+=arr[i];
                     }
                 }
                 return true;
             }
             long long findPages(int n, int arr[], int m) {
                 long long sum=0;
                 if(n<m)
                     return -1;
                 for(int i = 0; i < n; ++i)
                     sum += arr[i];
                 long long start=0,end=sum,mid;
                 long long int ans=int(1e15);
                 while(start<=end){
                     mid=(start+end)/2;
                     if(isPossible(arr,n,m,mid)){
                         ans=ans<mid?ans:mid;
                         end=mid-1;
                     }else{
                         start=mid+1;
                     }
                 }
                 return ans;
             }

### 3. Aggressive cows - Binary search
   - Initialize low as 1 (minimum possible distance) and high as the difference between the last and first stall (maximum possible distance).
   - Check if it’s possible to place all cows with a minimum distance of ‘mid’ using the ‘canPlaceCows’ function.
   - If the number of cows placed is equal to or greater than the required number of cows, return true.
   - If it’s possible to place all cows with the current minimum distance, increase the low to search for larger distances.
   - If it’s not possible, decrease the ‘high’ to search for smaller distances.
   -
           bool canPlaceCows(vector<int> &stalls, int distance, int cows) {
                int totalStalls = stalls.size();
                int cowsPlaced = 1; // Number of cows already placed
                int lastStall = stalls[0]; // Position of the last placed cow
                for (int i = 1; i < totalStalls; i++) {
                    // Check if the current stall can accommodate a cow
                    if (stalls[i] - lastStall >= distance) {
                        cowsPlaced++; // Place the next cow
                        lastStall = stalls[i]; // Update the position of the last placed cow
                    }
                    if (cowsPlaced >= cows) {
                        return true; // We can place all cows with the given distance
                    }
                }
                return false;
            }
            int findMaxMinDistance(vector<int> &stalls, int numberOfCows) {
                      int totalStalls = stalls.size();
                      sort(stalls.begin(), stalls.end());
                      int low = 1, high = stalls[totalStalls - 1] - stalls[0];
                      while (low <= high) {
                          int mid = low + (high - low) / 2;
                          if (canPlaceCows(stalls, mid, numberOfCows)) {
                              low = mid + 1;
                          } else {
                              high = mid - 1;
                          }
                      }
                      return high;
                  }

### 4. Minimum swaps to sort an array - Graph
   - We will have N nodes and an edge directed from node i to node j if the element at the i’th index must be present at the j’th index in the sorted array.
   - Sort curr and run a loop for i [0, N]
   - If the current element is already visited or it is at its correct position then continue
   - Else calculate the cycle size from the current element using a while loop
   - Declare an integer j equal to i and in the while loop set j equal to the index of curr[j] and increase cycle size while the element at jth position is not visited
   - Increase the answer by the size of the current cycle – 1 if the cycle size is more than 1.
   - 
              minSwaps(vector<int>&nums)
            	{
            	    int n=nums.size();
            	    vector<pair<int,int>> cur(n);
            	    for(int i=0;i<n;i++){
            	        cur[i].first=nums[i];
            	        cur[i].second=i;
            	    }
            	    sort(cur.begin(),cur.end());
            	    vector<bool> vis(n,false);
            	    int ans=0;
            	    for(int i=0;i<n;i++){
            	        if(vis[i] || cur[i].second==i){
            	            continue;
            	        }
            	        int cycle_size=0;
            	        int j=i;
            	        while(!vis[j]){
            	            vis[j]=true;
            	            j=cur[j].second;
            	            cycle_size++;
            	        }
            	        ans+=(cycle_size>1)?cycle_size-1:0;
            	    }
            	    return ans;
            	}


### 5. Search in a Rotated array - Binary search
   - Determine which side of the array is properly sorted:
   - If the left side (nums[low] to nums[mid]) is sorted: Check if the target lies within this range. If yes, adjust the high pointer to mid - 1. Otherwise, adjust the low pointer to mid + 1
   - If the right side (nums[mid] to nums[high]) is sorted: Check if the target lies within this range. If yes, adjust the low pointer to mid + 1. Otherwise, adjust the high pointer to mid - 1.
   -
              int search(vector<int>& nums, int target) {
                 int n=nums.size(),low=0,high=n-1;
                 while(low<=high){
                     int mid=(low+high)/2;
                     if(nums[mid]==target)
                         return mid;
                     if(nums[low]<=nums[mid]){
                         if(nums[low]<=target && nums[mid]>target)
                             high=mid-1;
                         else
                             low=mid+1;    
                     }else{
                         if(nums[high]>=target && nums[mid]<target)
                             low=mid+1;
                         else
                             high=mid-1; 
                     }   
                 }
                 return -1;
             }


## Day - 18

### 1. Count of small number  - Merge Sort
   - arr[i].second: This retrieves the original index of the element arr[i] in the nums array.
   - right - j + 1: This calculates the number of elements remaining in the right subarray starting from index j to right. Since the right subarray is already sorted and arr[i] is greater than arr[j], all these elements are smaller than arr[i].
   - count[arr[i].second] += (right - j + 1);: This updates the count of smaller elements to the right for the original index of arr[i] by adding the number of remaining elements in the right subarray
   - for (int l = left; l <= right; l++) This loop iterates over the indices from left to right, inclusive. The loop variable l represents the current index in the original array arr that we are updating with the merged values from temp
   -
                 void merge(int left,int mid,int right,vector<pair<int,int>> &arr,vector<int> &count){
                    vector<pair<int,int>> temp(right-left+1);
                    int i=left;
                    int j=mid+1;
                    int k=0;
                    while(i<=mid && j<=right){
                        if(arr[i].first<=arr[j].first)
                            temp[k++]=arr[j++];
                        else{
                            count[arr[i].second]+=(right-j+1);
                            temp[k++]=arr[i++];
                        }
                    }
                    while(i<=mid){
                        temp[k++]=arr[i++];
                    }
                    while(j<=right){
                        temp[k++]=arr[j++];
                    }
                    for(int l=left;l<=right;l++){
                        arr[l]=temp[l-left];
                    }
                }
                void mergeSort(int left,int right,vector<pair<int,int>> &arr,vector<int> &count){
                    if(left>=right)
                        return;
                    int mid=left+(right-left)/2;
                    mergeSort(left,mid,arr,count);
                    mergeSort(mid+1,right,arr,count);
                    merge(left,mid,right,arr,count);
                }
                vector<int> countSmaller(vector<int>& nums) {
                    int n=nums.size();
                    vector<pair<int,int>> arr;
                    for(int i=0;i<n;i++)
                        arr.push_back({nums[i],i});
                    vector<int> count(n,0);
                    mergeSort(0,n-1,arr,count);
                    return count;
                }

### 2. Split array largest sum  - Binary search
   - Initialize `low` to the maximum element in `nums` and `high` to the sum of all elements in `nums`.
   - Use binary search to find the minimum possible maximum subarray sum by adjusting `low` and `high` based on the number of subarrays needed.
   - In each iteration, calculate `mid` as the average of `low` and `high` and use the `check` function to determine how many subarrays are required if no subarray sum exceeds `mid`.
   - If the `check` function returns more subarrays than `k`, set `low` to `mid + 1`; otherwise, set `high` to `mid - 1`.
   - Continue the binary search until `low` exceeds `high`, then return `low` as the minimum possible maximum sum.
   -
                 int check(vector<int> &nums,int maxSum){
                    int count=1;
                    long long sum=0;
                    for(int i=0;i<nums.size();i++){
                        if(sum+nums[i]<=maxSum){
                            sum+=nums[i];
                        }else{
                            count++;
                            sum=nums[i];
                        }
                    }
                    return count;
                }
                int splitArray(vector<int>& nums, int k) {
                    int n=nums.size();
                    int low=0,high=0;
                    for(int i=0;i<n;i++){
                        high+=nums[i];
                        low=max(low,nums[i]);
                    }
                    while(low<=high){
                        int mid=(low+high)/2;
                        int count=check(nums,mid);
                        if(count>k){
                            low=mid+1;
                        }else{
                            high=mid-1;
                        }
                    }
                    return low;
                }

### 3. Smallest positive missing number - 
   - Segregate positive numbers from others i.e., move all non-positive numbers to the left side
   - Now ignore non-positive elements and consider only the part of the array which contains all positive elements.
   - Traverse the array containing all positive numbers and to mark the presence of an element x, change the sign of value at index x to negative.
   - Traverse the array again and print the first index which has a positive value.
   -
              int segregateArr(int arr[],int n){
                 int j=0,i;
                 for(int i=0;i<n;i++){
                     if(arr[i]<=0){
                         swap(arr[i],arr[j]);
                         j++;
                     }
                 }
                 return j;
             }
             int find(int arr[],int n){
                 for(int i=0;i<n;i++){
                     if(abs(arr[i])-1<n && arr[abs(arr[i])-1]>0)
                         arr[abs(arr[i])-1]=-arr[abs(arr[i])-1];
                 }
                 for(int i=0;i<n;i++){
                     if(arr[i]>0){
                         return i+1;
                     }
                 }
                 return n+1;
             }
             int missingNumber(int arr[], int n) 
             { 
                 int shift=segregateArr(arr,n);
                 return find(arr+shift,n-shift);
             } 


### 4. Middle of Linked List - Fast and slow pointers 
   - Initialize two pointers, fast and slow, both starting at the head of the linked list.
   - Move the fast pointer two steps at a time and the slow pointer one step at a time.
   - Continue moving the pointers until fast reaches the end of the list or fast's next node is null.
   - When the loop ends, the slow pointer will be at the middle node of the list.
   - Return the slow pointer as the middle node.
   -
           /**
             * Definition for singly-linked list.
             * struct ListNode {
             *     int val;
             *     ListNode next;
             *     ListNode() : val(0), next(nullptr) {}
             *     ListNode(int x) : val(x), next(nullptr) {}
             *     ListNode(int x, ListNode *next) : val(x), next(next) {}
             * };
             */
            class Solution {
            public:
                ListNode* middleNode(ListNode* head) {
                    ListNode *fast=head;
                    ListNode *slow=head;
                    while(fast!=NULL && fast->next!=NULL){
                        fast=fast->next->next;
                        slow=slow->next;
                    }
                    return slow;
                }
            };

### 5. Linked List Cycle - Fast and slow pointers 
   - Initialize two pointers, fast and slow, both starting at the head of the linked list.
   - Move the fast pointer two steps at a time and the slow pointer one step at a time.
   - Continue moving the pointers while checking if fast and fast->next are not null.
   - If fast and slow pointers meet, a cycle is detected, so return true.
   - If the loop ends without the pointers meeting, return false indicating no cycle is present
   -
           bool hasCycle(ListNode *head) {
              ListNode *fast=head,*slow=head;
              while(fast!=NULL && fast->next!=NULL){
                  fast=fast->next->next;
                  slow=slow->next;
                  if(fast==slow){
                      return true;
                  }
              }
              return false;
          }

### 6. Linked List Binary into integer - Math
   - Initialize an integer num to 0.
   - Traverse the linked list starting from the head node.
   - For each node, multiply num by 2 and add the node's value to num.
   - Move to the next node in the list and repeat the process.
   - Return num after the entire list has been processed.
   -
           int getDecimalValue(ListNode* head) {
              int num=0;
              while(head!=NULL){
                  num=num*2+head->val;
                  head=head->next;
              }
              return num;
          }

### 7. Linked List Binary into integer - List
   - Initialize a pointer res to the head of the linked list to keep track of the list's start.
   - Traverse the linked list using the head pointer while it and its next node are not null.
   - If the current node's value equals the next node's value, adjust the next pointer to skip the next node.
   - If the current node's value is not equal to the next node's value, move the head pointer to the next node.
   - Return the res pointer, which points to the head of the modified linked list without duplicates.
   -
           ListNode* deleteDuplicates(ListNode* head) {
              ListNode* res = head;
              while (head && head->next) {
                  if (head->val == head->next->val) {
                      head->next = head->next->next;
                  } else {
                      head = head->next;
                  }
              }
              return res;  
          }


### 8. Sort linked lists of 0s, 1s and 2s - List
   - The idea is to maintain 3 pointers named zero, one and two to point to current ending nodes of linked lists containing 0, 1, and 2 respectively. For every traversed node, we attach it to the end of its corresponding list.
        If the current node’s value is 0, append it after pointer zero and move pointer zero to current node.
        If the current node’s value is 1, append it after pointer one and move pointer one to current node.
        If the current node’s value is 2, append it after pointer two and move pointer two to current node.
   - Finally, we link all three lists. To avoid many null checks, we use three dummy pointers zeroD, oneD and twoD that work as dummy headers of three lists.
   -
              Node* sortList(Node* head) {
             if (!head || !(head->next)) 
                 return head; 
             Node* zeroD = new Node(0); 
             Node* oneD = new Node(0); 
             Node* twoD = new Node(0);
             Node *zero = zeroD, *one = oneD, *two = twoD;  
             Node* curr = head; 
             while (curr) { 
                 if (curr->data == 0) { 
                     zero->next = curr; 
                     zero = zero->next; 
                 } 
                 else if (curr->data == 1) { 
                     one->next = curr; 
                     one = one->next; 
                 } 
                 else { 
                     two->next = curr; 
                     two = two->next; 
                 } 
                 curr = curr->next; 
             } 
             zero->next = (oneD->next) ? (oneD->next) : (twoD->next); 
             one->next = twoD->next; 
             two->next = NULL; 
             // Updated head 
             head = zeroD->next; 
             // Delete dummy nodes 
             delete zeroD; 
             delete oneD; 
             delete twoD; 
             return head; 
         } 

### 9. Remove element from the LinkedList - Recursion
   - Create a dummy node temp with a value of 0 and set its next pointer to the head of the linked list.
   - Initialize a pointer curr to the dummy node temp.
   - Traverse the linked list using curr while curr->next is not null.
   - If the value of the next node equals val, update curr->next to skip the next node; otherwise, move curr to the next node.
   - Return temp->next, which points to the head of the modified linked list.
   -
              ListNode* removeElements(ListNode* head, int val) {
                 ListNode *temp=new ListNode(0);
                 temp->next=head;
                 ListNode *curr=temp;
                 while(curr->next!=NULL){
                     if(curr->next->val==val)
                         curr->next=curr->next->next;
                     else
                         curr=curr->next;
                 }
                 return temp->next;
             }

### 10. Merge tow LinkedLists - Recursion
   - Check if either list1 or list2 is null and return the non-null list if one is empty.
   - Compare the values of the current nodes in list1 and list2.
   - If list1's value is smaller, set list1->next to the result of merging the next node of list1 with list2, then return list1.
   - If list2's value is smaller or equal, set list2->next to the result of merging list1 with the next node of list2, then return list2.
   - Recursively merge the lists by repeating the above steps until both lists are fully merged
   -
              ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
                 if(list1==NULL || list2==NULL){
                     return list1?list1:list2;
                 }
                 if(list1->val<list2->val){
                     list1->next=mergeTwoLists(list1->next,list2);
                     return list1;
                 }else{
                     list2->next=mergeTwoLists(list1,list2->next);
                     return list2;
                 }
             }



## Day - 19

### 1. Multiply two linked lists  - Modular arithmetic
   - Convert the linked lists to integers: Traverse each linked list and construct the integer it represents.
   - Multiply the integers: Multiply the two integers obtained from the linked lists.
   - Take modulo 109+7109+7: Since the result can be very large, take the result modulo 109+7109+7.
   -
              long long multiplyTwoLists(Node *first, Node *second) {
                 long long num1 = 0, num2 = 0;
                 int mod = 1000000007;
                 while (first || second) {
                     if (first) {
                         num1 = ((num1 * 10) % mod + (first->data) % mod) % mod;
                         first = first->next;
                     }
                     if (second) {
                         num2 = ((num2 * 10) % mod + second->data % mod) % mod;
                         second = second->next;
                     }
                 }
                 return (num1 % mod * num2 % mod) % mod;
             }

### 2. Intersection of two linked lists  - Hash Table , Two pointers
   - Form a cycle in headA by connecting its tail to its head.
   - Detect the cycle using Floyd’s algorithm starting from headB.
   - Find the intersection by resetting one pointer to headB and moving both pointers step-by-step until they meet, then restore the list.
   -
           ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
              ListNode * tail=headA;
              while(tail->next)
                  tail=tail->next;
              tail->next=headA;
              ListNode *fast=headB,*slow=headB;
              while(fast && fast->next){
                  slow=slow->next;
                  fast=fast->next->next;
                  if(slow==fast){
                      slow=headB;
                      while(slow!=fast){
                          slow=slow->next;
                          fast=fast->next;
                      }
                      tail->next=NULL;
                      return slow;
                  }
              }
              tail->next=NULL;
              return NULL;
          }

### 3. Delete nodes having greater value on right  - Recursion
   - The base case of the recursion is when you reach the last node (head.next == null). In this case, there's nothing to compare with, so simply return the head node.
   - For each recursive call, the compute function is invoked on the next node (head.next) in the linked list. This effectively traverses the entire linked list in reverse order, reaching the end of the list first.
   - As the recursion unwinds, you start comparing each node (head) with its next node (head.next). If the value of the current node (head.data) is less than the value of its next node (head.next.data), it means the current node should be deleted since all nodes to its right are not greater than itself.
   - In such a case,  update the next pointer of the current node (head.next) to point directly to the result of the recursive call (compute(head.next)). This effectively skips the current node in the modified linked list.
   - If the value of the current node is greater than or equal to the value of its next node, you simply return the current node (head) since it remains valid in the new linked list.
   - The recursion continues until the first call, where the final modified linked list is returned.
   -
              Node *compute(Node *head)
             {
                 if (head->next == nullptr) {
                     return head;
                 }
                 head->next = compute(head->next);
                 if (head->data < head->next->data) {
                     return head->next;
                 }
                 return head;
             }

### 4. Palindrome Linked List  - Stack and queue
   -
              bool isPalindrome(ListNode* head) {
                 stack<int> start;
                 queue<int> end;
                 ListNode* curr = head;
                 while(curr != NULL){ 
                     start.push(curr->val); 
                     end.push(curr->val); 
                     curr = curr->next;
                 }
                 while(!start.empty() && !end.empty()){
                     if(start.top() != end.front()) return false;
                     start.pop(); end.pop();
                 }
                 return true;
             }

### 5. Reverse Linked List  - Recursion
   - Check if the current head is None. If it is, return None (base case).
   - Initialize newHead to the current head.
   - If the current head has a next node, make a recursive call to reverse the rest of the list.
   - After the recursive call returns, update the next pointer of the next node to point to the current head.
   - Break the original link by setting the next pointer of the current head to None.
   - Return newHead, which now points to the original last node, making it the new head of the reversed list.
   -
              ListNode* reverseList(ListNode* head) {
                 if (!head) {
                     return nullptr;
                 }
                 ListNode* newHead = head;
                 if (head->next) {
                     newHead = reverseList(head->next);
                     head->next->next = head;
                 }
                 head->next = nullptr;
                 return newHead;
             }


### 6. Add two numbers in Linked List - Recursion
   - Base Case: The function returns NULL if both lists are empty, or returns the non-empty list if one is NULL.
   - Sum Calculation: It creates a new node with the value of the sum of the current nodes' values from l1 and l2, modulo 10.
   - Carry Handling: If the sum is 10 or more, it adds a carry node (1) to the next recursive call, adjusting for the carry.
   -
           ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
              if (!l1 && !l2) 
                  return NULL;
              else if (!l1)
                  return l2;
              else if (!l2) 
                  return l1;
              int a = l1->val + l2->val;
              ListNode* p = new ListNode(a % 10);
              p->next = addTwoNumbers(l1->next, l2->next);
              if (a >= 10) 
                  p->next = addTwoNumbers(p->next, new ListNode(1));
              return p;
          }

     
### 7. Copy List with Random pointer - Hash Map
   - The basic idea is to traverse the list twice. In the first pass, we create a new node for each node in the original list and store the mapping in a hash map. In the second pass, we set the next and random pointers for each new node based on the hash map
   -
              Node* copyRandomList(Node* head) {
                 if(!head){
                     return NULL;
                 }
                 unordered_map<Node*,Node*> copy;
                 Node* curr=head;
                 while(curr){
                     copy[curr]=new Node(curr->val);
                     curr=curr->next;
                 }
                 curr=head;
                 while(curr){
                     copy[curr]->next=copy[curr->next];
                     copy[curr]->random=copy[curr->random];
                     curr=curr->next;
                 }
                 return copy[head];
             }


## Day - 20

### 1. Add two numbers in linked lists  - Stack
   - Sum and Carry: Initialize carry to 0 and create a dummy node ans; use temp to build the resulting list.
   - Adding Digits: Pop values from stacks, add them along with carry, create a new node for the current digit, and adjust carry.
   - Node Linking: Link the new node as the new head of the resulting list and update temp.
   - Final Carry Check: If there's a remaining carry, create a final node for it and link it as the head of the result list.
   -
              ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
                 stack<int> s1,s2;
                 while(l1){
                     s1.push(l1->val);
                     l1=l1->next;
                 }
                 while(l2){
                     s2.push(l2->val);
                     l2=l2->next;
                 }
                 int carry=0,sum=0;
                 ListNode* ans=new ListNode();
                 ListNode* temp=NULL;
                 while(!s1.empty() || !s2.empty()){
                     if(!s1.empty()){
                         sum+=s1.top();
                         s1.pop();
                     }
                     if(!s2.empty()){
                         sum+=s2.top();
                         s2.pop();
                     }
                     carry=sum/10;
                     ListNode* newhead=new ListNode(sum%10);
                     newhead->next=temp;
                     temp=newhead;
                     sum=carry;
                 }
                 if(carry!=0){
                     ListNode* newhead=new ListNode(carry);
                     newhead->next=temp;
                     temp=newhead;
                 }
                 return temp;
             }

### 2. Reverse Linked List Between - Swapping
   - Swap Node: Use swap to temporarily hold the node to be moved (swap = curr->next).
   - Update curr: Adjust curr->next to skip over the swap node (curr->next = swap->next).
   - Insert Node: Insert swap at the beginning of the reversed sublist (swap->next = prev->next and prev->next = swap).
   -
           ListNode* reverseBetween(ListNode* head, int left, int right) {
              if(head==NULL)
                  return NULL;
              if(head->next==NULL)
                  return head;
              ListNode* temp=new ListNode(0);
              temp->next=head;
              ListNode* prev=temp;
              for(int i=0;i<left-1;i++)
                  prev=prev->next;
              ListNode* curr=prev->next;
              for(int i=0;i<right-left;i++){
                  ListNode* swap=curr->next;
                  curr->next=swap->next;
                  swap->next=prev->next;
                  prev->next=swap;
              } 
              return temp->next;
          }

### 3. Reorder Linked List  - Stack
   - Find Middle: Use slow and fast pointers to locate the middle of the list, where slow ends up at the midpoint, and split the list into two halves.
   - Stack Push: Push nodes from the second half of the list onto a stack for later reversal.
   - Reorder List: Reattach nodes from the stack into the first half of the list, interleaving nodes from the stack with the remaining nodes.
   -
              void reorderList(ListNode* head) {
                 if(head==NULL || head->next==NULL) 
                     return ;
                 ListNode * dummy = new ListNode(0);
                 dummy->next = head;
                 ListNode * slow = dummy;
                 ListNode * fast = dummy;
                 while(fast && fast->next){
                     slow = slow ->next;
                     fast = fast->next->next;
                 }
                 stack <ListNode*> st;
                 ListNode* temp = slow->next;
                 slow -> next = NULL;
                 while(temp){
                     st.push(temp);
                     temp = temp->next;
                 }
                 slow = dummy->next;
                 while(!st.empty()){
                     temp = st.top();
                     st.pop();
                     temp->next = slow->next;
                     slow->next = temp;
                     slow = temp->next;
                 }
                dummy->next = NULL;
                delete dummy;
             }

### 4. Remove the Nth node from the end of the List  - Two pointers
   - Here we use a two-pointer technique to maintain a gap of n nodes between two pointers, ensuring that when the first pointer reaches the end, the second pointer is just before the node to be removed. This allows efficient removal of the nth node from the end in a single pass.
   - Advance Head: Move the head pointer n steps forward to maintain a gap of n nodes between head and ans.
   - Traverse List: Move both head and ans pointers one step at a time until head reaches the end of the list, keeping the gap constant.
   - Remove Node: Adjust the next pointer of the node pointed to by ans to skip the nth node from the end.
   -
              ListNode* removeNthFromEnd(ListNode* head, int n) {
                 ListNode* curr=new ListNode(0,head);
                 ListNode* ans=curr;
                 for(int i=0;i<n;i++){
                     head=head->next;
                 }
                 while(head){
                     head=head->next;
                     ans=ans->next;
                 }
                 ans->next=ans->next->next;
                 return curr->next;
             }

### 5. Flatten a mutilevel doubly linked List  - DFS
   - Traverse the list while handling child pointers:
      - Iterate through the list nodes.
      - If a node has a child, update the next and child pointers accordingly.
      - Collect the next node after the current level in a vector nodes (stack can also be used) for later connection.
   - Reverse Process Child Nodes:
      - Reverse the nodes vector to traverse child nodes in the correct order.
      - next pointer always indicate last node at end of each iteration. So that you can attach the next list's head to previous one's end
      - Connect each child node to its parent node.
               -
                 Node* flatten(Node* head) {
                    Node* curr=head;
                    Node* next;
                    vector<Node*> nodes;
                    while(curr){
                        Node* it=curr;
                        while(it->child==NULL && it->next!=NULL)
                            it=it->next;
                        Node* forward=it->next;
                        nodes.push_back(forward);
                        Node* ch=it->child;
                        it->next=ch;
                        it->child=NULL;
                        if(ch){
                            ch->prev=it;
                        }
                        next=it;
                        curr=ch;
                    }
                    reverse(nodes.begin(),nodes.end());
                    for(auto node:nodes){
                        Node* temp=node;
                        if(temp){
                            next->next=node;
                            node->prev=next;
                            while(temp->next!=NULL){
                                temp=temp->next;
                            }
                            next=temp;
                        }
                    }
                    return head;
                }


### 6. Partition List  - Two Pointers
   - Create two dummy nodes to serve as heads for the two partitions. Traverse the original linked list, and for each node
      - If the node's value is less than x, append it to the first partition.
      - If the node's value is greater than or equal to x, append it to the second partition.
   - Merge the two partitions by attaching the tail of the first partition to the head of the second partition.
   - Ensure to terminate the second partition's tail to prevent cycles in the resulting list
   -
              ListNode* partition(ListNode* head, int x) {
                 ListNode* newHead1 = new ListNode ( -1 ) ;
                 ListNode* tail1 = newHead1 ;
                 ListNode* newHead2 = new ListNode ( -1 ) ;
                 ListNode* tail2 = newHead2 ;
                 ListNode* ptr = head ;
                 while ( ptr != nullptr ) 
                 {
                     if ( ptr -> val < x )
                     {
                         tail1 -> next = ptr ;
                         ptr = ptr -> next ;
                         tail1 = tail1 -> next ;
                     }
                     else
                     {
                         tail2 -> next = ptr ;
                         ptr = ptr -> next ;
                         tail2 = tail2 -> next ;
                     }
                 }
                 tail1 -> next = newHead2 -> next ;
                 tail2 -> next = nullptr ;
                 return newHead1 -> next ;
             }
     
### 7. Remove duplicates from sorted list II  - Two Pointers
   - Use a dummy node to simplify edge cases.
   - Initialize two pointers, prev and curr. prev points to the last node in the result list, and curr traverses the list.
   - For each node, if it has duplicates, skip all nodes with the same value.
   - If no duplicates are found, move prev to the next node.
   -
              ListNode* deleteDuplicates(ListNode* head) {
                 if(head==NULL || head->next==NULL)
                     return head;
                 ListNode* dummy=new ListNode(0,head);
                 ListNode* prev=dummy;
                 ListNode* curr=head;
                 while(curr){
                     while(curr->next && curr->val==curr->next->val){
                         curr=curr->next;
                     }
                     if(prev->next!=curr){
                         prev->next=curr->next;
                     }else{
                         prev=prev->next;
                     }
                     curr=curr->next;
                 }
                 return dummy->next;
             }

### 8. Rearrange a Linked List in Zig-Zag fashion  - Linked List
   - Initialize Flag: Start with a boolean flag set to true, indicating that the next node should be greater in the desired zigzag pattern.
   - Traverse List: Iterate through the linked list, starting from the head, to the end of the list.
   - Check Relation: Depending on the flag, compare the current node's data with the next node's data:
       - If the flag is true and the current node's data is greater than the next node's data, swap their values.
       - If the flag is false and the current node's data is less than the next node's data, swap their values.
   - Advance and Toggle: Move to the next node and toggle the flag to indicate the opposite relation for the next pair of nodes
   -
              void zigZagList(Node* head)
               {
                   bool flag = true;
                   Node* current = head;
                   while (current->next != NULL) {
                       if (flag) /* "<" relation expected */
                       {
                           if (current->data > current->next->data)
                               swap(current->data, current->next->data);
                       }
                       else{
                           if (current->data < current->next->data)
                               swap(current->data, current->next->data);
                       }
                       current = current->next;
                       flag = !flag; /* flip flag for reverse checking */
                   }
               }

### 9. Sort Linked List - Merge sort
   - Merge Two Lists: Define mergeList to combine two sorted linked lists l1 and l2 by iterating through both lists, comparing values, and attaching the smaller value to the result list.
   - Base Case for Sorting: In sortList, return the head if the list is empty or has only one node, as it's already sorted.
   - Find Middle: Use slow and fast pointers to find the middle of the list, splitting it into two halves by setting the next of the node before the middle to NULL.
   - Recursive Sort: Recursively call sortList on each half to sort the two halves independently.
   - Merge Sorted Halves: Use mergeList to merge the two sorted halves back together and return the sorted list.
   -
              ListNode* mergeList(ListNode* l1,ListNode* l2){
                 ListNode* prev=new ListNode(-1),*curr=prev;
                 while(l1 && l2){
                     if(l1->val<=l2->val){
                         curr->next=l1;
                         l1=l1->next;
                     }else{
                         curr->next=l2;
                         l2=l2->next;
                     }
                     curr=curr->next;
                 }
                 if(l1){
                     curr->next=l1;
                     l1=l1->next;
                 }
                 if(l2){
                     curr->next=l2;
                     l2=l2->next;
                 }
                 return prev->next;
             }
             ListNode* sortList(ListNode* head) {
                 if(!head || !head->next)
                     return head;
                 ListNode* slow=head,*fast=head,*temp=NULL;
                 while(fast && fast->next){
                     temp=slow;
                     slow=slow->next;
                     fast=fast->next->next;
                 }
                 temp->next=NULL;
                 ListNode* point1=sortList(head),*point2=sortList(slow);
                 return mergeList(point1,point2);
             }

### 10. Segregate even and odd nodes in a Linked List - Linked list
   - split the linked list into two:  one containing all even nodes and the other containing all odd nodes. And finally, attach the odd node linked list after the even node linked list.
   -
               void segregateEvenOdd(struct Node** head_ref)
               {
                   Node* evenStart = nullptr;
                   Node* evenEnd = nullptr;
                   Node* oddStart = nullptr;
                   Node* oddEnd = nullptr;
                   Node* currNode = *head_ref;
                   while (currNode != nullptr) {
                       int val = currNode->data;
                       if (val % 2 == 0) {
                           if (evenStart == nullptr) {
                               evenStart = currNode;
                               evenEnd = evenStart;
                           }
                           else {
                               evenEnd->next = currNode;
                               evenEnd = evenEnd->next;
                           }
                       }
                       else {
                           if (oddStart == nullptr) {
                               oddStart = currNode;
                               oddEnd = oddStart;
                           }
                           else {
                               oddEnd->next = currNode;
                               oddEnd = oddEnd->next;
                           }
                       }
                       currNode = currNode->next;
                   }
                   if (oddStart == nullptr || evenStart == nullptr)
                       return;
                   evenEnd->next = oddStart;
                   oddEnd->next = nullptr;
                   *head_ref = evenStart;
               }

### 11. Rearrange a Linked List in place - Linked list
   - Find the middle point using tortoise and hare method.
   - Split the linked list into two halves using found middle point in step 1.
   - Reverse the second half.
   - Do alternate merge of first and second halves
   -
               void rearrange(Node** head)
               {
                   Node *slow = *head, *fast = slow->next;
                   while (fast && fast->next) {
                       slow = slow->next;
                       fast = fast->next->next;
                   }
                   Node* head1 = *head;
                   Node* head2 = slow->next;
                   slow->next = NULL;
                   reverselist(&head2);
                   *head = newNode(0); 
                   Node* curr = *head;
                   while (head1 || head2) {
                       if (head1) {
                           curr->next = head1;
                           curr = curr->next;
                           head1 = head1->next;
                       }
                       if (head2) {
                           curr->next = head2;
                           curr = curr->next;
                           head2 = head2->next;
                       }
                   }
                   *head = (*head)->next;
               }


### 12. Merge K sorted linked list - Heap priority queue
   - Check for Empty List: If the input lists is empty, return NULL as there are no lists to merge.
   - Priority Queue Initialization: Initialize a priority queue (pq) that will store pairs of integers and list indices. The queue is set up to use a min-heap to always provide the smallest value.
   - Populate Queue: Iterate through each linked list in lists. If a list is not empty (nullptr), push its first node's value and the index of the list into the priority queue.
   - Check Empty Queue: If the priority queue is still empty after the initial population, return NULL because there are no nodes to process.
   - Initialize Result List: Create a temporary dummy node temp to start building the result list. Set head to point to this dummy node to retain the head of the result list.
   - Process Queue: Enter a loop that continues until the priority queue is empty.
   - Extract Minimum: Inside the loop, extract the node with the smallest value from the priority queue. Assign this value to temp->val.
   - Advance List: Check if the extracted node's list has more nodes. If so, advance to the next node in that list and push its value and index into the priority queue.
   - Add New Node: If the priority queue is not empty, create a new node for the next position in the result list, and advance temp to this new node.
   - Break Loop: If the priority queue becomes empty, exit the loop. The result list is now fully constructed with head pointing to the first node of the merged sorted list. Return head.
   -
                 ListNode* mergeKLists(vector<ListNode*>& lists) {
                    if(lists.empty())
                        return NULL;
                    priority_queue<pair<int,int>,vector<pair<int,int>>,greater<pair<int,int>>> pq;
                    for(int i=0;i<lists.size();i++)
                    {
                        if(lists[i] != nullptr)
                        {
                            pq.push({lists[i]->val,i});
                        }
                    }
                    if(pq.empty())
                        return NULL;
                    ListNode*temp = new ListNode();
                    ListNode*head = temp;
                    while(true)
                    {
                        auto it = pq.top();
                        pq.pop();
                        temp->val = it.first;
                        if(lists[it.second]->next != nullptr)
                        {
                            lists[it.second] = lists[it.second]->next;
                            pq.push({lists[it.second]->val,it.second});
                        }
                        if(!pq.empty())
                        {
                            temp->next = new ListNode();
                            temp = temp->next;
                        }
                        else
                        {
                            break;
                        }
                    }
                    return head;
                }


## Day - 21

### 1. Reverse Nodes in K group  - Recursion
   - The program first checks if there are at least k nodes to reverse by iterating through the list and counting k nodes.
   - If there are at least k nodes, it reverses the next k nodes in the list using standard linked list reversal techniques.
   - After reversing k nodes, it recursively calls reverseKGroup on the remainder of the list to process the next group of k nodes.
   - If there are fewer than k nodes remaining, the function does not reverse these nodes and leaves them as they are.
   - Finally, the function connects the end of the reversed portion to the head of the next reversed segment (or remaining nodes) and returns the new head of the reversed list.
   -
                 ListNode* reverseKGroup(ListNode* head, int k) {
                    if(head==NULL){
                        return head;
                    }
                    ListNode *curr=head;
                    ListNode *temp=NULL;
                    ListNode *prev=NULL;
                    int count=0;
                    while(curr!=NULL && count<k){
                        curr=curr->next;
                        count++;
                    }
                    curr=head;
                    if(count==k){
                        count=0;
                        while(curr!=NULL && count<k){
                            temp=curr->next;
                            curr->next=prev;
                            prev=curr;
                            curr=temp;
                            count++;
                        }
                    }else{
                        prev=head;
                    }
                    if(temp!=NULL){
                        head->next=reverseKGroup(temp,k);
                    }
                    return prev;
                }

### 2. Subtraction in Linked List  - Linked List
   - Remove Leading Zeros: Ensure both linked lists do not have leading zeros (unless the number is zero).
   - Identify the Larger Number: Compare the two linked lists to determine which number is larger.
   - Perform Subtraction: Use recursion to traverse the lists to the end and subtract the smaller number from the larger number, managing the borrow operation.
   - Remove Leading Zeros from Result: Ensure the resulting list does not have leading zeros.
   -
                 Node* removeLeadingZeros(Node* head) {
                    while (head != nullptr && head->data == 0) {
                        head = head->next;
                    }
                    return head ? head : new Node(0);
                }
                int getLength(Node* head) {
                    int length = 0;
                    while (head != nullptr) {
                        length++;
                        head = head->next;
                    }
                    return length;
                }
                int compareLists(Node* head1, Node* head2) {
                    int len1 = getLength(head1);
                    int len2 = getLength(head2);
                    if (len1 != len2) {
                        return len1 - len2;
                    }
                    while (head1 != nullptr && head2 != nullptr) {
                        if (head1->data != head2->data) {
                            return head1->data - head2->data;
                        }
                        head1 = head1->next;
                        head2 = head2->next;
                    }
                    return 0;
                }
                int subtractHelper(Node* larger, Node* smaller, Node*& result) {
                    if (larger == nullptr) {
                        return 0;
                    }
                    Node* nextResult = nullptr;
                    int borrow = subtractHelper(larger->next, smaller ? smaller->next : nullptr, nextResult);
                    int largerVal = larger->data - borrow;
                    int smallerVal = smaller ? smaller->data : 0;
                    int diff = largerVal - smallerVal;
                    if (diff < 0) {
                        diff += 10;
                        borrow = 1;
                    } else {
                        borrow = 0;
                    }
                    result = new Node(diff);
                    result->next = nextResult;
                    return borrow;
                }
                Node* subLinkedList(Node* head1, Node* head2) {
                    head1 = removeLeadingZeros(head1);
                    head2 = removeLeadingZeros(head2);
                    Node* larger = head1;
                    Node* smaller = head2;
                    if (compareLists(head1, head2) < 0) {
                        larger = head2;
                        smaller = head1;
                    }
                    Node* result = nullptr;
                    subtractHelper(larger, smaller, result);
                    result = removeLeadingZeros(result);
                    return result;
                }

### 3. Flattening a Linked List  - Linked List
   - Define a vector (v) to store all the elements from the multi-level linked list.
   - Iterate through each level using two nested loops:
        - The outer loop iterates through each node on the top level (curr points to the current node).
        - The inner loop iterates through the bottom-linked list of the current node, adding each element to the vector (v).
   - After collecting all the elements in v, the code sorts the vector using the sort function.
   - A new linked list (head) is created, with the first element from the sorted vector as the initial node. Then, the remaining elements are added to the linked list using a loop:
        - A new node is created for each element in the sorted vector, and it is linked to the previous node's bottom.
        - This effectively constructs a single-level linked list containing all the elements from the multi-level linked list.
   - The final flattened and sorted linked list is returned
   -
                 Node *flatten(Node *root) {
                   vector<int>v;
                   Node *curr = root;
                   while(curr!=NULL){
                       Node *temp = curr;
                       while(temp!=NULL){
                           v.push_back(temp->data);
                           temp = temp->bottom;
                       }
                       curr=curr->next;
                   }
                   sort(v.begin(), v.end());
                   Node *head = new Node(v[0]);
                   Node *curr1 = head;
                   for(int i=1;i<v.size();i++){
                       curr1->bottom = new Node(v[i]);
                       curr1 = curr1->bottom;
                   }
                   return head;
                }


### 4. Implementing queue using stack  - Stack design
   - Push Operation (push): Use one stack (stack1) for enqueueing elements by pushing them onto the stack.
   - Pop and Peek Operations (pop and peek): For dequeueing elements, use a second stack (stack2) as an auxiliary stack. When performing a pop or peek operation, check if stack2 is empty. If it is, transfer elements from stack1 to stack2 to reverse the order. The top element of stack2 will then represent the front of the queue.
   - Transfer Elements (transferElements): This function transfers elements from stack1 to stack2. While stack1 is not empty, pop elements from stack1 and push them onto stack2. This step is crucial for maintaining the FIFO order when using two stacks.
   -
              class MyQueue {
               public:
                   stack<int> s1,s2;
                   void transferEle(){
                       while(!s1.empty()){
                           s2.push(s1.top());
                           s1.pop();
                       }
                   }
                   MyQueue() {
                   }
                   void push(int x) {
                       s1.push(x);
                   }
                   int pop() {
                       if(s2.empty()){
                           transferEle();
                       }
                       int front=s2.top();
                       s2.pop();
                       return front;
                   }
                   int peek() {
                       if(s2.empty()){
                           transferEle();
                       }
                       return s2.top();
                   }
                   bool empty() {
                       return s1.empty() && s2.empty();
                   }
               };


### 5. Backspace string compare  - Two Pointers
   - Initialize two pointers, i and j, to the last indices of strings s and t, respectively. Also, initialize two variables, skipS and skipT, to keep track of the number of backspaces encountered for each string.
   - Use a while loop to iterate as long as at least one of the strings has characters left to process (i >= 0 || j >= 0).
   - For each string s and t, find the next character that is not a backspace ('#'):
        - While i >= 0 and s[i] is a backspace ('#'), increment skipS and decrement i.
        - While j >= 0 and t[j] is a backspace ('#'), increment skipT and decrement j.
   - Compare the current characters in s and t at indices i and j. If they are not equal and both i and j are valid indices, return false.
   - If one of the strings has reached its end while the other hasn't, return false (e.g., if i >= 0 is not equal to j >= 0).
   -
           bool backspaceCompare(string s, string t) {
              int i=s.length()-1,j=t.length()-1;
              int skipS=0,skipT=0;
              while(i>=0 || j>=0){
                  while(i>=0 && (s[i]=='#' || skipS>0)){
                      if(s[i]=='#'){
                          skipS++;
                      }else{
                          skipS--;
                      }
                      i--;
                  }
                  while (j >= 0 && (t[j] == '#' || skipT > 0)) {
                      if (t[j] == '#') {
                          skipT++;
                      } else {
                          skipT--;
                      }
                      j--;
                  }
                  if((i>=0 && j>=0) && s[i]!=t[j]){
                      return false;
                  }
                  if ((i >= 0) != (j >= 0)) {
                      return false;
                  }
                  i--;
                  j--;
              }
              return true;
          }

### 6. Implement stack using queues  - Queue design
   - When pushing a new element, we first push it onto the queue. Then, we rotate the queue by pushing and popping elements until the newly pushed element becomes the front.
   - Pop, top, and empty operations are straightforward, as they directly operate on the single queue
   -
                 class MyStack {
                  public:
                      queue<int> q;
                      MyStack() {
                      }
                      void push(int x) {
                          q.push(x);
                          int n=q.size();
                          for(int i=0;i<n-1;i++){
                              q.push(q.front());
                              q.pop();
                          }
                      }
                      int pop() {
                          int result=q.front();
                          q.pop();
                          return result;
                      }
                      int top() {
                          return q.front();
                      }
                      bool empty() {
                          return q.empty();
                      }
                  };

### 7. Implement stack and queues using Dequeue  - Doubly linked list
   - https://www.geeksforgeeks.org/implement-stack-queue-using-deque/

### 8. Next greater element on the right - Hash and stack
   - The function initializes a result vector res with the same size as nums1, filled with -1, a stack st, and a map mp.
   - It iterates over each element num in nums2, and for each element, it updates the map with the next greater element for elements stored in the stack.
   - During the iteration, if the current element num is greater than the stack's top element, it maps the stack's top element to num and pops the stack.
   - After processing nums2, it iterates over each element in nums1, updating res with the corresponding next greater element from the map if it exists.
   - Finally, the function returns the result vector res containing the next greater elements for each element in nums1
   -
              vector<int> nextGreaterElement(vector<int>& nums1, vector<int>& nums2) {
                 vector<int> res(nums1.size(),-1);
                 stack<int> st;
                 map<int,int> mp;
                 for(int num:nums2){
                     while(!st.empty() && st.top()<num){
                         mp[st.top()]=num;
                         st.pop();
                     }
                     st.push(num);
                 }
                 for(int i=0;i<nums1.size();i++){
                     if(mp[nums1[i]]){
                         res[i]=mp[nums1[i]];
                     }
                 }
                 return res;
             }


## Day - 22

### 1. Evaluation of Postfix Expression - Stack
   - If the character is an operator (+, -, *, /), we pop the top two elements from the stack.
   - We perform the operation indicated by the character using the two operands.
   - We push the result of the operation back onto the stack.
   -
              int evaluatePostfix(string S)
             {
                 stack<int> st;
                 for (char c : S) {
                     if (isdigit(c)) {
                         st.push(c - '0');
                     } else {
                         int operand2 = st.top();
                         st.pop();
                         int operand1 = st.top();
                         st.pop();
                         switch (c) {
                             case '+':
                                 st.push(operand1 + operand2);
                                 break;
                             case '-':
                                 st.push(operand1 - operand2);
                                 break;
                             case '*':
                                 st.push(operand1 * operand2);
                                 break;
                             case '/':
                                 st.push(operand1 / operand2);
                                 break;
                         }
                     }
                 }
                 return st.top();
             }

### 2. Implement two stacks in an array - Stack
   - Stack1 starts from the leftmost corner of the array, the first element in stack1 is pushed at index 0 of the array. 
   - Stack2 starts from the rightmost corner of the array, the first element in stack2 is pushed at index (n-1) of the array. 
   - Both stacks grow (or shrink) in opposite directions. 
   -
              class twoStacks {
                 public:
                   int *arr;
                   int size;
                   int top1,top2;
                   twoStacks(int n=100) {
                       size=n;
                       arr=new int[n];
                       top1=-1;
                       top2=n;
                   }
                   void push1(int x) {
                       if(top1<top2-1){
                           top1++;
                           arr[top1]=x;
                       }
                   }
                   void push2(int x) {
                       if(top1<top2-1){
                           top2--;
                           arr[top2]=x;
                       }
                   }
                   int pop1() {
                       if(top1>=0){
                           int x=arr[top1];
                           top1--;
                           return x;
                       }else
                           return -1;
                   }
                   int pop2() {
                       if(top2<size){
                           int x=arr[top2];
                           top2++;
                           return x;
                       }else
                           return -1;
                   }
               };
     

### 3. Daily Temperature - Monotonic Stack
   - We traverse the temperatures array from right to left, maintaining a stack that stores the indices of the elements. For each element, we pop elements from the stack until we find an element greater than the current one. The difference in indices gives us the number of days until the next warmer day.
   -
              vector<int> dailyTemperatures(vector<int>& temperatures) {
                 stack<int> st;
                 int n=temperatures.size();
                 vector<int> wait(n,0);
                 for(int i=n-1;i>=0;i--){
                     while(!st.empty() && temperatures[i]>=temperatures[st.top()])
                         st.pop();
                     if(!st.empty())
                         wait[i]=st.top()-i;
                     st.push(i);
                 }
                 return wait;
             }

### 4. Minimum cost tree for Leaf values - Dynammic programming, Monotonic Stack
   - Since the non-leaf node value is the product of previous largest leaf and new leaf, the problem can be considered as a monostack problem.
   - In order to find PGE (Previous Greater Element), the stack should be monotonous decreasing stack.
   - Iterate Over Array: For each element in the array, maintain a leaf variable to track the smallest popped value.
   - Monotonic Stack Maintenance: Use a while loop to pop elements from the stack if they are less than the current element, adding their products to ans and updating leaf.
   - Push Current Element: Add the product of leaf and the current element to ans and push the current element onto the stack.
   - Finalize Result: After processing the array, compute the sum of products for remaining elements in the stack to complete the result.
   -
     int mctFromLeafValues(vector<int>& arr) {
        int n=arr.size(),ans=0;
        vector<int> st;
        for(int i=0;i<n;i++){
            int leaf=0;
            while(!st.empty() && st.back()<arr[i]){
                ans+=leaf*st.back();
                leaf=st.back();
                st.pop_back();
            }
            ans+=leaf*arr[i];
            st.push_back(arr[i]);
        }
        for(int i=0;i<st.size()-1;++i){
            ans+=st[i]*st[i+1];
        }
        return ans;
    }

### 5. Online Stock span - Monotonic Stack
   - Pop from Stack: While the stack is not empty and the current price is greater than or equal to the price at the index stored at the top of the stack, pop the stack.
   - Calculate Span: If the stack is empty after popping, the span is the length of the vector plus one; otherwise, it is the difference between the current index and the index at the top of the stack.
   - Push to Stack and Vector: Push the current index onto the stack and the current price onto the vector, then return the calculated span.
   -
                 class StockSpanner {
            public:
                stack<int> st;
                vector<int> v;
                StockSpanner() {}
                int next(int price) {
                    int span;
                    while(!st.empty() && price>=v[st.top()]) 
                        st.pop();
                    span = st.empty() ? v.size()+1 : v.size()-st.top();
                    st.push(v.size());
                    v.push_back(price);    
                        
                    return span;     
                }
            };

### 6. Rotten oranges - Queue, BFS
   - Create an empty queue Q and two variables count(store number of oranges which is rotten or needs to be rotten) and answer(store answer). 
   - Find all rotten oranges and enqueue them to Q also set the count variable accordingly. Run a loop While Q is not empty.
   - Increase the answer by 1.
   - Now calculate the size of the queue(i.e number of elements present in the queue) and run a loop from 0 to the size of the queue.
         - Dequeue an orange from the queue, and rot all adjacent oranges(if present) and decrease the count variable.
         - While rotting the adjacent, push the index into the queue.
   - Now the oranges rotten this time frame is only present in the queue. If count>0 then return -1.
   -
              int orangesRotting(vector<vector<int>>& grid) {
                 int count=0,res=-1;
                 queue<vector<int>> q;
                 vector<vector<int>> dir={{-1,0},{1,0},{0,-1},{0,1}};
                 for(int i=0;i<grid.size();i++){
                     for(int j=0;j<grid[0].size();j++){
                         if(grid[i][j]>0){
                             count++;
                         }
                         if(grid[i][j]==2)
                             q.push({i,j});
                     }
                 }
                 while(!q.empty()){
                     res++;
                     int size=q.size();
                     for(int k=0;k<size;k++){
                         vector<int> cur=q.front();
                         count--;
                         q.pop();
                         for(int i=0;i<4;i++) 
                         {
                             int x=cur[0]+dir[i][0], y=cur[1]+dir[i][1];
                             if(x>=grid.size()||x<0||y>=grid[0].size()||y<0||grid[x][y]!=1) 
                                 continue;
                             grid[x][y]=2;
                             q.push({x, y});
                         }
                     }
                 }
                 if(count==0)
                     return max(0,res);
                 else
                     return -1;
             }

### 7. Sum of Subarray minimum - Monostatic stack,DP
   - If the stack is not empty after popping, the element arr[i] is the minimum for subarrays starting from the index of the element at the top of the stack (stack.back()) to i.
   - Compute the contribution of arr[i] as dp[j] + arr[i] * (i - j) where j is the index at the top of the stack. Here, arr[i] contributes to the subarrays formed from j+1 to i.
   - If the stack is empty, arr[i] is the minimum for all subarrays starting from index 0 to i, so compute the contribution as arr[i] * (i + 1).
   -
              int sumSubarrayMins(vector<int>& arr) {
                 int n=arr.size(), mod=1e9+7;
                 vector<long long> dp(n, -1);
                 vector<int> stack;
                 long long ans=0;
                 for(int i=0;i<n;i++){
                     while(!stack.empty() && arr[i]<=arr[stack.back()])
                         stack.pop_back();
                     if (!stack.empty()){
                         int j=stack.back();
                         dp[i]=dp[j]+arr[i]*(i-j);
                     }
                     else 
                         dp[i]=arr[i]*(i+1);
                     stack.push_back(i);
                     ans=(ans+dp[i])%mod;
                 }
                 return ans;
             }

### 8. Evaluate reverse polish notation - Stack
   - If the token is an operator:
        - Pop the top two elements from the stack.
        - Perform the operation based on the operator.
        - Push the result back onto the stack.
   - If the token is an operand: Convert it to an integer and push it onto the stack.
   -
              int evalRPN(vector<string>& tokens) {
                 int n=tokens.size();
                 stack<int> st;
                 for(int i=0;i<n;i++){
                     if(tokens[i] == "+" or tokens[i] == "-" or tokens[i] == "*" or tokens[i] == "/"){
                             int num1 = st.top(); st.pop();
                             int num2 = st.top(); st.pop();
                             int newNum = 0;
                             if(tokens[i] == "+") newNum = num2+num1;
                             else if(tokens[i] == "-") newNum = num2-num1;
                             else if(tokens[i] == "*") newNum = num2*num1;
                             else newNum = num2/num1;
                             st.push(newNum);
                     }
                     else st.push(stoi(tokens[i]));
                 }
                 return st.top();
             }



## Day - 23

### 1. Circular tower - Sliding window, Two pointers, Queue
   - Set two pointers, start = 0 and end = 1 to use the array as a queue.
        - If the amount of petrol is efficient to reach the next petrol pump then increment the end pointer (circularly).
        - Otherwise, remove the starting point of the current tour, i.e., increment the start pointer.
   - If the start pointer reaches N then such a tour is not possible. Otherwise, return the starting point of the tour.
   -
           int tour(petrolPump p[],int n)
          {
             int start=0,end=1;
             int curr=p[start].petrol-p[start].distance;
             while(end!=start || curr<0){
                 while(curr<0 && start!=end){
                     curr-=p[start].petrol-p[start].distance;
                     start=(start+1)%n;
                     if(start==0)
                      return -1;
                 }
                 curr+=p[end].petrol-p[end].distance;
                 end=(end+1)%n;
             }
             return start;
          }

### 2. Remove all adjecent duplicated in a string II - Stack
   - For each character in s, it pushes the character and a count of 1 onto the stack if the stack is empty or the character is different from the stack's top character.
   - If the character is the same as the stack's top character, it increments the count at the stack's top, and pops the stack if the count reaches k.
   - After processing all characters, the function constructs the result string by concatenating characters from the stack based on their counts.
   -
               string removeDuplicates(string s, int k) {
                  stack<pair<char,int>> st;
                  string res="";
                  for(int i=0;i<s.size();i++){
                      if(st.empty() || s[i]!=st.top().first){
                          st.push({s[i],1});
                      }else{
                          st.top().second++;
                          if(st.top().second==k){
                              st.pop();
                          }
                      }
                  }
                  while(!st.empty()){
                      res=string(st.top().second,st.top().first)+res;
                      st.pop();
                  }
                  return res;
              }

### 3. Flatten nested list iterator - Stack, Tree, DFS
   - In the constructor, the flatten function is called with the provided nestedList. This function recursively processes the nested structure and extracts integers from it.
   - During the initialization phase, the flatten function populates the flattened vector with integers found in the nested list.
   - The iterator keeps track of the current index, currentIndex, to indicate the next element to be retrieved using the next function.
   - The next function returns the element at the current index and increments the index to prepare for the next call.
   - The hasNext function checks if there are more elements to be processed by comparing the currentIndex with the size of the flattened vector.
   -
               class NestedIterator {
                public:
                    vector<int> flat;
                    int curr;
                    void flatten(const vector<NestedInteger> &nestedList){
                        for(const auto& item:nestedList){
                            if(item.isInteger()){
                                flat.push_back(item.getInteger());
                            }else{
                                flatten(item.getList());
                            }
                        }
                    }
                    NestedIterator(vector<NestedInteger> &nestedList) {
                        flatten(nestedList);
                        curr=0;
                    }
                    int next() {
                        return flat[curr++]; 
                    }
                    bool hasNext() {
                        return curr<flat.size();
                    }
                };


### 4. Maximum of Minimum of Every window size - Stack, Sliding window
   - Initializing ans array of size n+1 with all values as 0. ans[i] will keep track of maximum of minimum values in window of size i.Filling answer list  by comparing minimums of all lengths computed using left[] and right[].
   - Run a loop for i from 0 to n-1:
   - In each iteration for loop: For ith element find length len for which it will be minimum element and take take max of ans[len] and a[i] and store the result in ans[len]
   - The result for length i, i.e. ans[i] would always be greater or same as result for length i+1, i.e., ans[i+1].
   - If ans[i] is not filled it means there is no direct element which is minimum of length i and therefore either the element of length ans[i+1], or ans[i+2], and so on is same as ans[i].
   - Run a loop for i from n-1 to 1, and in each iteration do ans[i] = max(ans[i], ans[i + 1]).
   -
             vector <int> maxOfMin(int arr[], int n)
            {
                stack<int> s;
                int left[n+1];
                int right[n+1];
                for(int i=0;i<n;i++){
                    left[i]=-1;
                    right[i]=n;
                }
                for(int i=0;i<n;i++){
                    while(!s.empty() && arr[s.top()]>=arr[i])
                        s.pop();
                    if(!s.empty())
                        left[i]=s.top();
                    s.push(i);
                }
                while(!s.empty())
                    s.pop();
                for(int i=n-1;i>=0;i--){
                    while(!s.empty() && arr[s.top()]>=arr[i])
                        s.pop();
                    if(!s.empty())
                        right[i]=s.top();
                    s.push(i);
                }
                int ans[n+1];
                for(int i=0;i<=n;i++)
                    ans[i]=0;
                for(int i=0;i<n;i++){
                    int len=right[i]-left[i]-1;
                    ans[len]=max(ans[len],arr[i]);
                }
                for (int i = n - 1; i >= 1; i--) 
                    ans[i] = max(ans[i], ans[i + 1]);
                vector<int> res(n);
                for(int i=1;i<=n;i++){
                    res[i-1]=ans[i];
                }
                return res;
            }

### 5. LRU Cache - Hash table, doubly linked list
   - Put
    I find it in my hashmap if it is not present and size of unorderd map is less than container then I create a newNode right to head node of doubly linked list and insert map[key]=newNode.
    continer is full then I delete the Node from Linked List left to tail delete(tail->prev). and then do same insertion as point 1.
    if it is availabe in map then we make it most recent used we delete it from its location and put it right to head(head->next=NewNode) and update key value with newNode.
   - Get
    For get we just find the key in hashmap and if it is not present then return -1. elseif it is present then delete it from its location make it most recently used right to head and return its value
   -
             class LRUCache {
            public:
                int cap;
                struct Node{
                    int key;
                    int data;
                    struct Node *prev,*next;
                    Node(int x,int y){
                        key=x;
                        data=y;
                        prev=NULL;
                        next=NULL;
                    }
                };
                unordered_map<int,Node*> mp;
                Node *head=new Node(0,0);
                Node *tail=new Node(0,0);
                LRUCache(int capacity) {
                    cap=capacity;
                    head->next=tail;
                    tail->prev=head;
                }
                Node* createNode(int key,int val)
                {
                  Node*newNode=new Node(key,val);
                    newNode->next=head->next;
                        head->next->prev=newNode;
                        head->next=newNode;
                        newNode->prev=head;
                        return newNode;
                }
                void deleteNode(Node* x)
                {
                  Node*temp=x;
                  temp->prev->next=temp->next;
                  temp->next->prev=temp->prev;
                  temp->next=NULL;
                  temp->prev=NULL;
                  delete temp; 
                }
                int get(int key) {
                    if(mp.find(key)!=mp.end()){
                        auto it=mp.find(key);
                        int ans=it->second->data;
                        Node *newNode=createNode(it->second->key,it->second->data);
                        deleteNode(it->second);
                        mp.erase(key);
                        mp[key]=newNode;
                        return newNode->data;
                    }
                    return -1;
                }
                void put(int key, int value) {
                    if(mp.find(key)==mp.end())
                   {
                     if(mp.size()<cap)
                     {
                        Node*newNode=createNode(key,value);
                        mp[key]=newNode;
                      }
                      else
                      {
                          Node*x=tail->prev;
                          int y=x->key;
                          deleteNode(x);
                          mp.erase(y);
                         Node*newNode=createNode(key,value);
                        mp[key]=newNode;
                      }
                   }
                   else
                   {
                     auto it=mp.find(key);
                     Node*x=it->second;
                          int y=x->key;
                          deleteNode(x);
                          mp.erase(y);
                           Node*newNode=new Node(key,value);
                        newNode->next=head->next;
                        head->next->prev=newNode;
                        head->next=newNode;
                        newNode->prev=head;
                        mp[key]=newNode;
                   } 
                }
            };

### 6. Celebrity Problem - Stack, Two pointers
   - Initialize Pointers: Start with two pointers: a at the beginning (0) and b at the end (n-1).
   - If M[a][b] is 1, person a knows b, so a cannot be a celebrity, and move the a pointer to a + 1.
   - If M[a][b] is 0, person a does not know b, so b cannot be a celebrity, and move the b pointer to b - 1.
   - Check if this potential celebrity (a) is known by everyone and knows no one else. For a = 1: Check if M[1][i] = 0 for all i ≠ 1 (person 1 should know no one else). Check if M[i][1] = 1 for all i ≠ 1 (everyone should know person 1).
   -
             int celebrity(vector<vector<int> >& mat) {
                int n=mat.size();
                int a=0,b=n-1;
                while(a<b){
                    if(mat[a][b])
                        a++;
                    else
                        b--;
                }
                for(int i=0;i<n;i++){
                    if((i!=a) && (mat[a][i] || !mat[i][a]))
                        return -1;
                }
                return a;
            }

### 7. Diameter of Binary Tree - DFS
   - If the current node is null, it returns a height of 0.
   - It calculates the height of the left and right subtrees by recursively calling height.
   - It updates the diameter to the maximum value between the current diameter and the sum of the heights of the left and right subtrees.
   -
           int height(TreeNode* root,int &diameter){
              if(!root)
                  return 0;
              int l=height(root->left,diameter);
              int r=height(root->right,diameter);
              diameter=max(diameter,l+r);
              return max(l,r)+1;
          }
          int diameterOfBinaryTree(TreeNode* root) {
              int diameter=0;
              height(root,diameter);
              return diameter;
          }

### 8. Invert Binary Tree - DFS
   -
           TreeNode* invertTree(TreeNode* root) {
              if(!root)
                  return NULL;
              TreeNode* save=root->left;
              root->left=root->right;
              root->right=save;
              invertTree(root->right);
              invertTree(root->left);
              return root;
          }
