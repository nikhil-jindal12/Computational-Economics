"""
project 7. the Solow-Swan growth model
"""
# Section 1. Preparation. Import the necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import copy


# Section 2. Define the Growth model as a class

# ==================
# attributes:
# -a: factor productivity
# -s: saving rate
# -alpha: share of capital
# -delta: depreciation rate
# -n: population growth

# -k: capital per capita
# -K: aggregate capital
# -y: income per capita
# -Y: aggregate income
# -i: investment per capita
# -I: aggregate investment
# -L: population
# -d: depreciation
# -be: break-even investment
# -steady_state: steady state values


# methods:
# check_model: print the attribute of the instance
# growth: take one argument "years", and drive the economic growth
# get_param: get the model parameters
# get_state: get the state variables
# plot_growth: visualize how the income per capita, investment per capita, and capital accumulate
# plot_income_growth: visualize income growth over time
# find_steady_state: solve for the steady state in this model

# arguments:
# ----------------------------------------------------------------------------
# para_dict: dictionary of parameters

# ----------------------------------------------------------------------------
# state_dict: dictionary of state variables


class Growth_Model:
    """
    This class will create an instance of the Solow-Swan growth model
    
    arguments:
    ------------------
    para_dict: dict
        a dictionary of parameters
    state_dict: dict
        a dictionary of model state
        
    attributes
    --------------
    
    methods:
    -------------
    
    """
    
    def __init__(self, para_dict, state_dict):
        # read-in the given parameters and variables
        self.para_dict = para_dict
        self.state_dict = state_dict
        
        # calculate lower case y
        self.state_dict['y'] = (self.para_dict['a'][0]
                                * self.state_dict['k']**self.para_dict['alpha'][0])
        # calculate upper case K (i.e., aggregate capital)
        self.state_dict['K'] = self.state_dict['k'] * self.state_dict['L']
        # calculate upper case Y
        self.state_dict['Y'] = self.para_dict['a'][0] * self.state_dict['k']**self.para_dict['alpha'][0] * self.state_dict['L']**(1-self.para_dict['alpha'][0])
        # calculate lower case i
        self.state_dict['i'] = self.para_dict['s'][0] * self.state_dict['y']
        # calculate the upper case I
        self.state_dict['I'] = self.para_dict['s'] * self.state_dict['Y']
        
        self.steady_state = {}
        
    def get_param(self, key):
        return self.para_dict[key]
    
    def get_state(self, key):
        return self.state_dict[key]
    
    
    def growth(self, years):
        # step 1. define the time line
        time_line = np.linspace(0, years, num=years+1, dtype=int)

        # step 2. examine growth
        for t in time_line: 
            
            # 2.1. load parameters
            n = self.para_dict.get('n')[0]
            s = self.para_dict.get('s')[0]
            alpha = self.para_dict.get('alpha')[0]
            delta = self.para_dict.get('delta')[0]
            a = self.para_dict.get('a')[0]
            
            # 2.2. load all current states
            y_t = self.state_dict.get('y')
            k_t = self.state_dict.get('k')
            Y_t = self.state_dict.get('Y')
            L_t = self.state_dict.get('L')
            K_t = self.state_dict.get('K')
            i_t = self.state_dict.get('i')
            I_t = self.state_dict.get('I')

            # 2.3 calculate new states i.e., the dynamic
            dk = s * y_t - (delta + n)*k_t
            k_next = k_t + dk
            y_next = a * k_next**alpha
            K_next = k_next*L_t
            Y_next = a * (K_next**alpha) * (L_t**(1-alpha))
            i_next = y_next*s
            I_next = i_next*L_t

            # update the attributes
            self.state_dict['k'] = k_next
            self.state_dict['y'] = y_next
            self.state_dict['Y'] = Y_next
            self.state_dict['K'] = K_next
            self.state_dict['L'] = L_t
            self.state_dict['i'] = i_next
            self.state_dict['I'] = I_next
        
    def find_steady_state(self):
        '''
        when dk is negative, not in 
        when dk is pos, then have passed steady state
        want to make dk as close to 0 as possible
        '''
        
        # step 1. load parameters
        n = self.para_dict.get('n')[0]
        s = self.para_dict.get('s')[0]
        alpha = self.para_dict.get('alpha')[0]
        delta = self.para_dict.get('delta')[0]
        a = self.para_dict.get('a')[0]
        
        # store the results
        k_star = (s * a / (n + delta) ** (1 / (1-alpha)))
        y_star = a * (k_star ** alpha)
        i_star = s * y_star
        c_star = y_star - i_star
        
        steady_state = {}
        steady_state["k_star"] = k_star
        steady_state["y_star"] = y_star
        steady_state["c_star"] = c_star
        steady_state["i_star"] = i_star
        
        self.steady_state = steady_state
        
        return steady_state

    def plot_income_growth(self, ax, years):
        # plot income growth over time
        timeline = np.arange(0, years + 1)
        income_data = []
        income_data.append(self.state_dict['y'])
        
        for x in range(years):
            self.growth(1)
            income_data.append(self.state_dict['y'])
        
        ax.plot(timeline, income_data, label='Income per Capita')
        ax.set_title('Income Growth over Time')
        ax.set_xlabel('Time (yrs)')
        ax.set_ylabel('Income per Capita')
        ax.legend()
    
    def plot_growth(self, ax):
        # plot_growth(): visualize the relationship between 
        #    income per capita, investment per capita, and capital accumulate. 
        # (i.e., income per capita & investment per capita against capital ) 
        k_vals = np.linspace(0.1, 10, 100)
        s = self.para_dict.get('s')[0]
        alpha = self.para_dict.get('alpha')[0]
        a = self.para_dict.get('a')[0]
        delta = self.para_dict.get('delta')[0]
        n = self.para_dict.get('n')[0]
        
        y_vals = a * (k_vals ** alpha)
        i_vals = s * y_vals  
        break_even_vals = (delta + n)*(k_vals)
        
        ax.plot(k_vals, y_vals, label="Income per Capita")
        ax.plot(k_vals, i_vals, label="Investment per Capita")
        ax.plot(k_vals, break_even_vals, label="Break Even Point")
        
        ax.set_title("Income and Investment per Capita vs. Capital per Capita")
        ax.set_xlabel("Capital per Capita")
        ax.set_ylabel("Per Capita Values")
        ax.legend()


# Section 3. Specify model parameters and examine economic growth
# set parameters (exogenously given):
parameters = {'n': np.array([0.002]),                 # population growth rate
              's': np.array([0.15]),                  # saving rate
              'alpha': np.array([1/3]),               # share of capital
              'delta': np.array([0.05]),              # depreciation rate
              'a': np.array([1])                      # technology 
              }

states = {              # factor productivity
          'k': np.array([1]),
          'L': np.array([100])}


# instantiate a growth model
model = Growth_Model(parameters, states)

steady_states = model.find_steady_state()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
model.plot_income_growth(ax1, 100)
model.plot_growth(ax2)
plt.show()

# Section 4

steady_state_consumption_33 = {'n': np.array([0.002]), 
                               's': np.array([0.33]), 
                               'alpha': np.array([1/3]), 
                               'delta': np.array([0.05]), 
                               'a': np.array([1])}
model_33 = Growth_Model(steady_state_consumption_33, states)
print(f"c* at 33% saving rate: {model_33.find_steady_state()['c_star']}")

steady_state_consumption_50 = {'n': np.array([0.002]),
                               's': np.array([0.5]),
                               'alpha': np.array([1/3]),
                               'delta': np.array([0.05]),
                               'a': np.array([1])}
model_50 = Growth_Model(steady_state_consumption_50, states)
print(f"c* at 50% saving rate: {model_50.find_steady_state()['c_star']}")