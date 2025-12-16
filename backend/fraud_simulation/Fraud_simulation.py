#%%
from Network import run_simulation
import matplotlib.pyplot as plt

# RUN & SAVE DATASET

N_USERS = int(input("Enter number of users in the network: "))
N_SIMULATION_HOURS = int(input("Enter number of hours to simulate: "))
BASE_FRAUD_RATE = float(input("Enter base fraud rate (e.g. 0.01): "))

df, acc = run_simulation(N_USERS, N_SIMULATION_HOURS, BASE_FRAUD_RATE)
df.to_csv("bhim_upi_fraud_simulated_randomized.csv", index=False)

print("Dataset saved: bhim_upi_fraud_simulated.csv")
print(df.head())

# PERFORMANCE PLOT

plt.figure(figsize=(10,4))
plt.plot(acc)
plt.title("Online Fraud Model – Balanced Accuracy")
plt.xlabel("Transaction Index")
plt.ylabel("Balanced Accuracy")
plt.show()
