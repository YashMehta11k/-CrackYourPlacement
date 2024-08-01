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
