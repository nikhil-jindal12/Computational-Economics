import numpy as np
import matplotlib.pyplot as plt

# Section 1 - Set the model parameters

total_pop = 3500
initial_infected = int(total_pop * 0.01)
initial_recovered = int(total_pop * 0)
transmission_rate = 0.8
recovery_rate = 0.05

# Section 2 - Define the initial status table

sir = np.zeros((1,3))
sir[0, 0] = total_pop - initial_infected - initial_recovered
sir[0, 1] = initial_infected
sir[0, 2] = initial_recovered

# Section 3 - Simulate the epidemic of the virus

sir_sim = sir.copy()
susceptible_pop_norm = [] 
infected_pop_norm = [] 
recovered_pop_norm = [] 
days = 100
total_days = np.linspace(0, days, num=days)

for day in total_days:
    # calculate new infected number
    new_infected = int(transmission_rate * (sir_sim[0,0]/total_pop) * sir_sim[0, 1])
    
    # set total susceptible people as the upper limit of new infected
    if (new_infected > sir_sim[0,0]):
        new_infected = sir_sim[0,0]
    
    # define new recovered number
    new_recovered = int(sir_sim[0,1] * recovery_rate)
    
    # remove new infections from susceptible group
    sir_sim[0,0] = sir_sim[0,0] - new_infected
    
    # add new infections into infected group, 
    sir_sim[0,1] = sir_sim[0,1] + new_infected
    
    # also remove recovers from the infected group
    sir_sim[0,1] = sir_sim[0,1] - new_recovered
    
    # add recovers to the recover group
    sir_sim[0,2] = sir_sim[0,2] + new_recovered
    
    # set lower limits of all the groups (0 people)   
    for group in sir_sim[0]:
        # print(group)
        if group < 0:
            group = 0
    
    # normalize the SIR (i.e., % of population) and append to the record lists
    susceptible_pop_norm.append(sir_sim[0,0]/total_pop)
    infected_pop_norm.append(sir_sim[0,1]/total_pop)
    recovered_pop_norm.append(sir_sim[0,2]/total_pop)
        
    
outcome = [susceptible_pop_norm, infected_pop_norm, recovered_pop_norm]

# Section 4 - Plot the results

# define the plot function
def sir_simulation_plot(outcome,days):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(1,1,1)
    days = np.linspace(1,days,num=days)
    susceptible = np.array(outcome[0])*100
    infected = np.array(outcome[1])*100
    recovered = np.array(outcome[2])*100
    ax.plot(days,susceptible,label='susceptible',color='y')
    ax.plot(days,infected,label='infected',color='r')
    ax.plot(days,recovered,label='recovered',color='g')
    ax.set_xlabel('Days')
    ax.set_ylabel('Proportion of the population')
    ax.set_title("SIR Model Simulation")
    plt.legend()
    plt.show()
# call the function to plot the outcome    
sir_simulation_plot(outcome,days=days)