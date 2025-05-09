# SARSA Learning Algorithm
## AIM
To develop a Python program to find the optimal policy for the given RL environment using SARSA-Learning and compare the state values with the Monte Carlo method.
## PROBLEM STATEMENT

# Step1 :
Set Q-values to zero for all state-action pairs. Prepare ε-greedy policy and decay schedules for ε and α.

# Step2:
For each episode, interact with the environment using ε-greedy policy. Update Q-values using SARSA rule: 𝑄 ( 𝑠 , 𝑎 ) ← 𝑄 ( 𝑠 , 𝑎 ) + 𝛼 ⋅ [ 𝑟 + 𝛾 𝑄 ( 𝑠 ′ , 𝑎 ′ ) − 𝑄 ( 𝑠 , 𝑎 ) ]

# Step3:
After all episodes, derive π(s) = argmaxₐ Q(s,a). Compute value function V(s) = maxₐ Q(s,a) for each state.
## SARSA LEARNING FUNCTION
# Name:Himavath M
# Reg no:212223240053
## OUTPUT:
![image](https://github.com/user-attachments/assets/694560bd-22f8-4b63-bfbb-a6152806f33d)
![image](https://github.com/user-attachments/assets/a4ce3071-b3f7-41ca-992c-cc144a592039)
![image](https://github.com/user-attachments/assets/883ecfc3-6517-4b1a-bf42-5085e2b2fb41)
![image](https://github.com/user-attachments/assets/561fa473-54ff-4fe8-a519-0988f764bc03)
![image](https://github.com/user-attachments/assets/ef989fd7-3dd6-4049-a219-51168c69c799)


## RESULT:
Thus, The Python program to find the optimal policy for the given RL environment using SARSA-Learning is executed successfully.
