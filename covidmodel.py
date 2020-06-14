from scipy import interpolate
from datetime import datetime as dt
import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# Function that defines the model in terms of compartment derivatives

def derivatives(y, t, R0_st, k, x0, R0_end, gamma, sigma, N, p_I_to_C, p_C_to_D, Beds):
    S, E, I, C, R, D = y
    def b1(t): ##
        return I / (I + C) * (12 * p_I_to_C + 1/gamma * (1 - p_I_to_C)) + C / (I + C) * (
                    min(Beds(t), C) / (min(Beds(t), C) + max(0, C-Beds(t))) * (p_C_to_D * 7.5 + (1 - p_C_to_D) * 6.5) + 
                    max(0, C-Beds(t)) / (min(Beds(t), C) + max(0, C-Beds(t))) * 1 * 1
                             )
    def beta(t):
        return log_R0(t, R0_st, k, x0, R0_end) / b1(t) if not np.isnan(b1(t)) else 0

    # print(beta(t))
    dSdt = -beta(t) * I * S / N
    # print(I, S/N, -beta(t), dSdt)
    dEdt = beta(t) * I * S / N - sigma * E
    dIdt = sigma * E - 1/12.0 * p_I_to_C * I - gamma * (1 - p_I_to_C) * I
    dCdt = 1/12.0 * p_I_to_C * I - 1/7.5 * p_C_to_D * min(Beds(t), C) - max(0, C-Beds(t)) - (1 - p_C_to_D) * 1/6.5 * min(Beds(t), C)
    dRdt = gamma * (1 - p_I_to_C) * I + (1 - p_C_to_D) * 1/6.5 * min(Beds(t), C)
    dDdt = 1/7.5 * p_C_to_D * min(Beds(t), C) + max(0, C-Beds(t))
    return dSdt, dEdt, dIdt, dCdt, dRdt, dDdt


gamma = 1.0/9.0
sigma = 1.0/5.1


def log_R0(t, R0_st, k, x0, R0_end):
    return (R0_st-R0_end) / (1 + np.exp(-k*(-t+x0))) + R0_end


# init_cases = threshold number of cases
# init_date = date the init_cases are met
# N = population
# beds_per_1000k = total number of beds
# R0_st = 2 
# k = 0.5
# x0 = date of lockdown
# R0_end = 
# prob_Inf_to_Crit = 5%
# prob_Crit_to_D = 6.5%
# s = 0.003


def simulate(country, params):
    total_beds, R0_st, k, x0, R0_end = params
    # country = "germany"
    # Read in the CSV files of dataset as dataframes
    census_df = pd.read_csv("data\\" + country + "\\census.csv")
    cases_df = pd.read_csv("data\\" + country + "\\cases.csv")
    deaths_df = pd.read_csv("data\\" + country + "\\deaths.csv")

    # Process the dataframes to extract the relevant information
    if country == 'ireland':
        func = lambda x: "-".join(x.split()[0].split('/'))
        # func = lambda x: "-".join(x.split('/')[::-1])
    elif country == 'germany':
        func = lambda x: "-".join(x.split('/')[::-1])
    elif country == 'italy' or country == 'uk':
        func = lambda x: "-".join([x.split('/')[2], ('0' + x.split('/')[0])[-2:], ('0' + x.split('/')[1])[-2:]])
    dates_cases = np.array(list(map(func, cases_df["date"].values)))
    dates_deaths = np.array(list(map(func, deaths_df["date"].values)))
    confirmed_cases = cases_df["sum_cases"].values.astype(np.int32)
    confirmed_deaths = deaths_df["sum_deaths"].values.astype(np.int32)

    # Extract total population of the country
    population = int(census_df["Total"].sum()) 

    init_cases = confirmed_cases[0]
    init_date = dates_cases[0]

    N = population
    prob_Inf_to_Crit = 0.05
    prob_Crit_to_D = 0.065
    s = 0.003

    def Beds(t):
        beds_0 = total_beds  # stores beds per 100 k -> get total number
        return beds_0 + s*beds_0*t  # 0.003
        
    def CovidModel(init_cases, init_date, N, total_beds, R0_st, k, x0, R0_end, prob_Inf_to_Crit, prob_Crit_to_D, s, r0_y_interpolated=None):
        days = int(1500)

        y0 = N-init_cases, 0.0, init_cases, 0.0, 0.0, 0.0 # no one exposed yet 
        t = np.linspace(0, days, days)
        print(y0)
        ret = odeint(derivatives, y0, t, args=(R0_st, k, x0, R0_end,
                                            gamma, sigma, N, prob_Inf_to_Crit, prob_Crit_to_D, Beds))
        S, E, I, C, R, D = ret.T
        # R0_over_t = r0_y_inter
        total_CFR = [0] + [100 * D[i] / sum(sigma*E[:i]) if sum(
            sigma*E[:i]) > 0 else 0 for i in range(1, len(t))]
        daily_CFR = [0] + [100 * ((D[i]-D[i-1]) / ((R[i]-R[i-1]) + (D[i]-D[i-1]))) if max(
            (R[i]-R[i-1]), (D[i]-D[i-1])) > 10 else 0 for i in range(1, len(t))]
        
        dates = pd.date_range(start=np.datetime64(init_date), periods=days, freq="D")

        return dates, S, E, I, C, R, D, total_CFR, daily_CFR, [Beds(i) for i in range(len(t))]

    # Compute the curves by using the optimized parameters
    dates, S, E, I, C, R, D, total_CFR, daily_CFR, beds = CovidModel(
                                cases_df["sum_cases"][0], 
                                init_date, 
                                N=population, 
                                total_beds=total_beds, 
                                R0_st=R0_st, 
                                k=k, 
                                x0=x0, 
                                R0_end=R0_end, 
                                prob_Inf_to_Crit=prob_Inf_to_Crit, 
                                prob_Crit_to_D=prob_Crit_to_D, 
                                s=s)

    dates = list(map(lambda date: str(date.date()), dates))
    return dates, S.tolist(), E.tolist(), (I*0.2).tolist(), I.tolist(), C.tolist(), R.tolist(), D.tolist(), (E+I+C).tolist(), total_CFR, daily_CFR, beds, confirmed_cases.tolist(), confirmed_deaths.tolist()



def get_optimized_params(country):
    # country = "germany"
    # Read in the CSV files of dataset as dataframes
    census_df = pd.read_csv("data\\" + country + "\\census.csv")
    cases_df = pd.read_csv("data\\" + country + "\\cases.csv")
    deaths_df = pd.read_csv("data\\" + country + "\\deaths.csv")
    beds_df = pd.read_csv("data\\" + country + "\\beds.csv", header=None)

    # Process the dataframes to extract the relevant information
    if country == 'ireland':
        func = lambda x: "-".join(x.split()[0].split('/'))
    elif country == 'germany':
        func = lambda x: "-".join(x.split('/')[::-1])
    elif country == 'italy' or country == 'uk':
        func = lambda x: "-".join([x.split('/')[2], ('0' + x.split('/')[0])[-2:], ('0' + x.split('/')[1])[-2:]])
    dates_cases = np.array(list(map(func, cases_df["date"].values)))
    dates_deaths = np.array(list(map(func, deaths_df["date"].values)))
    confirmed_cases = cases_df["sum_cases"].values.astype(np.int32)
    confirmed_deaths = deaths_df["sum_deaths"].values.astype(np.int32)

    # If dates of cases and deaths are inconsistent, give a warning to the user
    if (dates_cases != dates_deaths).any():
        print("-" * 30)
        print("WARNING: dates for confirmed cases and deaths don't match.")
        print("-" * 30)

    # Extract total population of the country
    population = int(census_df["Total"].sum()) 

    # Extract total number of beds as the last value before an empty cell
    index = beds_df.index[beds_df[8].isnull()][1] - 1
    total_beds = int(beds_df.values[index, 8])
    if country == 'ireland':
        total_beds = (1-0.9)*total_beds
    elif country == 'germany':
        total_beds = (1-0.798)*total_beds
    elif country == 'italy':
        total_beds = (1-0.789)*total_beds
    elif country == 'uk':
        total_beds = (1-0.843)*total_beds
    total_beds = int(total_beds)

    init_cases = confirmed_cases[0]
    # "DD/MM/YYYY" to "YYYY-MM-DD"
    init_date = dates_cases[0]

    N = population
    prob_Inf_to_Crit = 0.05
    prob_Crit_to_D = 0.065
    s = 0.003

    # Default values for parameters to be optimized
    R0_st = 2.4
    k = 0.5
    x0 = 60
    R0_end = 1
    defaults = (R0_st, k, x0, R0_end)

    regularization = 0.

    def Beds(t):
        beds_0 = total_beds  # stores beds per 100 k -> get total number
        return beds_0 + s*beds_0*t  # 0.003

    def residuals_totalcases(params):
        # Unpack the params which is in form of tuple
        R0_st, k, x0, R0_end = params
        # R0_st, k, x0, R0_end = min(max(R0_st, 0), 5.), min(max(k, 0.1), 5), max(x0, 0), max(min(R0_end, R0_st), 0)
        
        y0 = N-init_cases, 0.0, init_cases, 0.0, 0.0, 0.0 # no one exposed yet 
        t = np.linspace(0, len(confirmed_cases), len(confirmed_cases))

        ret = odeint(derivatives, y0, t, args=(R0_st, k, x0, R0_end,
                                            gamma, sigma, N, prob_Inf_to_Crit, prob_Crit_to_D, Beds))
        S, E, I, C, R, D = ret.T

        # weights = list(range(1, len(confirmed_cases) + 1))
        # Apply loss as sum of squared differences
        ratio = (I + E + C) / (np.maximum(D, 1) * 3)

        residuals_value = sum(
            ((I + E + C) - confirmed_cases) ** 2
            + ((D - confirmed_deaths) ** 2)
        ) + regularization * sum((param1)**2 for param1, param2 in zip(params, defaults))
        return residuals_value

    # TO DO: Change these bounds for testing out different results
    bounds_params = (
        (2, 20), (0.01, 10), (0, len(dates_cases)), (0.4, 10)
    )

    # Parameters to be optimized
    optimize_params = (R0_st, k, x0, R0_end)

    # Perform optimization
    output = minimize(
        residuals_totalcases,
        optimize_params,
        # method='trust-constr',
        bounds=bounds_params,
        options={'maxiter': 1000000, 'verbose': 0}
    )

    # Get the updated parameters
    R0_st, k, x0, R0_end = output.x

    return total_beds, R0_st, k, x0, R0_end


######### covid_data_totalz.groupby("Date").sum()[["Value"]].plot(figsize=(12, 8), title="Covid-19 total fatalities (world)");
if __name__ == "__main__":

    process_country("germany")
    process_country("ireland")