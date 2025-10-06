#!/usr/bin/env python3
"""
Greedy Algorithm for Container Allocation Optimization

This module provides a class-based implementation of the greedy algorithm
for container-to-barge allocation optimization.
"""

import random
import math
import networkx as nx
import numpy as np


class GreedyOptimizer:
    """Greedy algorithm for container allocation optimization"""
    
    def __init__(self, qk=None, h_b=None, h_t_40=200, h_t_20=140, handling_time=1/6):
        """Initialize the greedy optimizer with default or custom parameters"""
        self.Qk = qk if qk is not None else [104, 99, 81, 52, 28]  # TEU
        self.H_b = h_b if h_b is not None else [3700, 3600, 3400, 2800, 1800]  # euros
        self.H_t_40 = h_t_40  # euros
        self.H_t_20 = h_t_20  # euros
        self.Handling_time = handling_time  # hours (10 minutes)
        
        # These will be set by generate_instance or load_instance
        self.C_dict = {}
        self.C = 0
        self.N = 0
        self.T_matrix = []
        self.Barges = []
        self.C_ordered = []
        
        # Results storage
        self.f_ck_init = None
        self.route_list = []
        self.barge_departure_delay = []
        self.trucked_containers = {}
        self.total_cost = 0
    
    @property
    def T_ij_list(self):
        """Alias for T_matrix for backward compatibility"""
        return self.T_matrix

# Global variables for backward compatibility
Qk = [104, 99, 81, 52, 28] # TEU
H_b = [3700, 3600, 3400, 2800, 1800] # euros
H_t_40 = 200 # euros
H_t_20 = 140 # euros
Handling_time = 1/6 # hours (10 minutes)

def container_info(seed, reduced):
    random.seed(seed)

    if reduced:
        C = random.randint(33,200) # number of containers
        N = random.randint(4,8) # number of terminals
    else:
        C = random.randint(100,600) # number of containers
        N = random.randint(10,20) # number of terminals

    C_dict = {}

    for i in range(C):
        
        Dc = random.randint(24,196) # closing hours
        Oc = random.randint(Dc-120,Dc-24) # opening hours
        P_40 = random.uniform(0.75,0.9) # probability of 40ft container
        P_Export = random.uniform(0.05,0.7) # probability of export

        if random.random() < P_40:
            Wc = 2 # 40ft container
        else:
            Wc = 1 # 20ft container
        
        if random.random() < P_Export:
            In_or_Out = 2 # Export
            Rc = random.randint(0,24)
            Terminal = random.randint(1,N-1) # assigned delivery terminal location
        else:
            In_or_Out = 1 # Import
            Rc = 0
            Terminal = random.randint(1,N-1) # assigned pickup terminal location
        
        C_dict[i] = {
            "Rc": Rc,
            "Dc": Dc,
            "Oc": Oc,
            "Wc": Wc,
            "In_or_Out": In_or_Out,
            "Terminal": Terminal
        }

    return C_dict, C, N

seed = 100 # you can change the seed for different instances
Reduced = False # set to True for reduced instances, False for full instances

[C_dict, C, N] = container_info(seed, Reduced)

# print(C_dict) # dictionary of containers with their attributes

# Example output
# print(C_dict[56]["In_or_Out"]) # 1 for Import, 2 for Export

def T_ij(N):
    T_ij_list = np.zeros((N, N), dtype=int)

    number_of_sub_terminals = math.floor((N - 1) / 3)

    index_antwerp = list(range(1,number_of_sub_terminals+1)) # Antwerp sub-terminals
    index_rotterdam = list(range(number_of_sub_terminals+1, 2*number_of_sub_terminals+1)) # Rotterdam sub-terminals
    index_maasvlakte = list(range(2*number_of_sub_terminals+1,N)) # Maasvlakte sub-terminals

    if len(index_maasvlakte) == number_of_sub_terminals+2: # making sure the length of Maasvlakte is not too different to other two
        index_rotterdam.append(index_maasvlakte[0])
        index_maasvlakte = index_maasvlakte[1:]

    # print(index_antwerp, index_rotterdam, index_maasvlakte)

    for i in range(N):
        for j in range(N):
            if i == j:
                T_ij_list[i][j] = 0

            elif i in index_antwerp and j in index_antwerp:
                T_ij_list[i][j] = 1
            elif i in index_rotterdam and j in index_rotterdam:
                T_ij_list[i][j] = 1
            elif i in index_maasvlakte and j in index_maasvlakte:
                T_ij_list[i][j] = 1
                
            elif (i == 0 or j == 0) and (j in index_antwerp or i in index_antwerp):
                T_ij_list[i][j] = 13

            elif (i == 0 or j == 0) and ((j in index_maasvlakte or j in index_rotterdam) or (i in index_maasvlakte or i in index_rotterdam)):
                T_ij_list[i][j] = 11

            elif (i in index_antwerp or j in index_antwerp) and ((j in index_maasvlakte or j in index_rotterdam) or (i in index_maasvlakte or i in index_rotterdam)):
                T_ij_list[i][j] = 16

            elif (i in index_maasvlakte or j in index_maasvlakte) and (i in index_rotterdam or j in index_rotterdam):
                T_ij_list[i][j] = 4
    
            else:
                T_ij_list[i][j] = 666            

    return T_ij_list

T_ij_list = T_ij(N)
n = len(T_ij_list)

# Example output
# print(T_ij_list[1][5]) # 1 for Antwerp sub-terminal, 5 for Rotterdam sub-terminal

# Master route

G = nx.complete_graph(n)
for i in range(n):
    for j in range(n):
        if i != j:
            G[i][j]['weight'] = T_ij_list[i][j]

# Find approximate TSP cycle (returns to start)
master_route = nx.approximation.traveling_salesman_problem(G, cycle=True, weight='weight')

#print("Master route:", master_route) # used to order containers for the greedy algorithm for the inital solution

C_ordered = [] 

for i in master_route[1:-1]:
    for j in range(C):
        if C_dict[j]["Terminal"] == i:
            C_ordered.append(j)

# print("Ordered containers:", C_ordered) # ordered list of containers based on the master route

Barges = [104, 99, 81, 52, 28] # TEU

def get_route(L_current):

    route = [0]  # start at terminal 0 (main terminal)
    current_terminal = 0 # start at terminal 0 (main terminal)

    for c in L_current.values():
        if c["Terminal"] == current_terminal:
            continue
        else:
            route.append(c["Terminal"])
            current_terminal = c["Terminal"]
    
    return route

def get_timing(route, T_ij_list, handling_time, L_current, delay):

    departure_time = 0

    current_max = 0
    dry_port_handling_time = 0

    for c in L_current.values():
        if c["In_or_Out"] == 2:
            dry_port_handling_time += handling_time

            if c["Rc"] > current_max:
                current_max = c["Rc"]
                
    departure_time = current_max + dry_port_handling_time

    D_terminal = [departure_time]  # time of departure from each terminal
    O_terminal = [0]  # time of arrival at each terminal

    current_terminal = 0  # start at terminal 0 (dry port)

    term_departure_time = departure_time  # start time at departure time
    term_arrival_time = 0  # start time at 0

    for terminal in route[1:]:
        
        travel_time = T_ij_list[current_terminal][terminal]
        handling_time_total = handling_time * sum(1 for c in L_current.values() if c["Terminal"] == terminal)

        term_departure_time += travel_time # add travel time to next terminal
        term_departure_time += handling_time_total # add handling time at terminal

        term_arrival_time += travel_time + D_terminal[-1]  # arrival time is the same as departure time after handling

        D_terminal.append(term_departure_time)  # append the time of arrival at the terminal
        O_terminal.append(term_arrival_time)  # append the time of arrival at the terminal

        current_terminal = terminal
    
    D_terminal[-1] += delay  # add delay to the last terminal's departure time

    return D_terminal, O_terminal

def check_for_cap(route, L_current, barges, idx):

    teu_used = []

    for i in range(len(route)):
        Terminal = route[i] 
        sum = 0
        for c in L_current.values():
            if c["Terminal"] == Terminal and Terminal > 0 and c["In_or_Out"] == 1: # import containers are loaded on the barge
                sum += c["Wc"]
            elif c["Terminal"] == Terminal and Terminal > 0 and c["In_or_Out"] == 2: # export containers are unloaded from the barge
                sum -= c["Wc"]
            elif i == 0 and c["In_or_Out"] == 2:
                sum += c["Wc"]

        teu_used.append(sum)

    cap = [True if teu <= barges[idx] else False for teu in teu_used]

    if all(cap):
        return True
    else:
        return False

def delay_window(container, D_terminal, route, terminal, handling_time):

    Oc = container["Oc"]

    D_term = D_terminal[route.index(terminal)]

    if Oc - D_term > 0:
        delay = (Oc - D_term) + handling_time
        return delay
    else:
        return 0
    
f_ck_init = np.zeros((C, len(Barges)))  # matrix for container to barge assignment, if container c is assigned to barge b, f_ck_init[c][b] = 1
 
barge_idx = 0
to_ignore = []  # list to store containers that can be removed from C_ordered
departure_delay = 0       # ← carry this forward across containers
barge_departure_delay = []
route_list = []

while barge_idx < len(Barges):
    for c in C_ordered:
        if c in to_ignore:
            continue

        # 1) Tentatively assign c to this barge
        f_ck_init[c, barge_idx] = 1

        # 2) Build the current load
        L_current = {
            cont: C_dict[cont]
            for cont in C_ordered
            if f_ck_init[cont, barge_idx] == 1
        }
        route = get_route(L_current)

        # 3) Capacity check
        if not check_for_cap(route, L_current, Barges, barge_idx):
            f_ck_init[c, barge_idx] = 0
            continue

        # 4) Time‐window check, with up to one “departure shift”
        success = False
        delay = departure_delay  # start from whatever delay we already have

        # Try once to adjust departure (max_tries = 1)
        for attempt in range(2):      # attempt = 0 (no shift), attempt = 1 (shift)
            D_term, O_term = get_timing(route, T_ij_list, Handling_time, L_current, delay)

            # find any containers that now violate
            violations = []
            for cont in L_current.values():
                t = cont["Terminal"]
                if not (
                    O_term[route.index(t)] <= cont["Oc"] <= D_term[route.index(t)]
                    or O_term[route.index(t)] <= cont["Dc"] <= D_term[route.index(t)]
                ):
                    violations.append(cont)

            if not violations:
                # everyone fits under this `delay`
                success = True
                break

            # if we still have our one “shift” left, compute the shift
            if attempt == 0:
                # largest extra wait needed
                needed = [
                    delay_window(v, O_term, route, v["Terminal"], Handling_time)
                    for v in violations
                ]
                delay += max(needed)  # accumulate shift
            else:
                # second pass and still violations → fail
                break

        if success:
            # commit this shift permanently for the rest of this barge
            departure_delay = delay
            to_ignore.append(c)
        else:
            # undo assignment
            f_ck_init[c, barge_idx] = 0

    # move on to next barge (with its own fresh departure_delay)
    route_list.append(route)
    barge_departure_delay.append(departure_delay)
    barge_idx += 1
    departure_delay = 0

index_to_be_trucked = np.where(np.sum(f_ck_init, axis=1) == 0)[0].tolist()

trucked_containers = {i: C_dict[i] for i in index_to_be_trucked}

truck_cost = 0

for i in trucked_containers:
    if C_dict[i]["Wc"] == 1: # 20ft container
        truck_cost += H_t_20
    else:  # 40ft container
        truck_cost += H_t_40

xijk = np.zeros((len(Barges), N, N))  # xijk[i][j][k] = 1 if barge k goes from terminal i to terminal j

for barge_idx, route in enumerate(route_list):
    for i in range(len(route) - 1):
        xijk[barge_idx][route[i]][route[i + 1]] = 1
        xijk[barge_idx][route[i + 1]][route[i]] = 1

def objective(x, H_b, T, N):
    K = len(H_b)         # number of barges
    cost = 0

    for k in range(K):
        # 1) fixed‐cost term: sum over j≠0 of x[0][j][k]*H_b[k]
        for j in range(N):
            if j == 0:
                continue
            cost += x[k][0][j] * H_b[k]

        # 2) travel‐time term: sum over all i,j of T[i][j]*x[i][j][k]
        for i in range(N):
            for j in range(N):
                cost += T[i][j] * x[k][i][j]

        # 3) extra‐stop term: sum over j≠0, i≠j of x[j][i][k]
        for j in range(N):
            if j == 0:
                continue
            for i in range(N):
                if i == j:
                    continue
                cost += x[k][i][j] * Handling_time

    return cost

barge_cost = objective(xijk, H_b, T_ij_list, N)

total_cost = barge_cost + truck_cost

print("Total cost:", total_cost)