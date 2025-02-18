import random
import math
import numpy as np
import matplotlib.pyplot as plt
import heapq

# ==============================
# Problem 1: Generate Exponential Random Variables
# ==============================
def generate_exponential(lambda_val):
    """
    Generate an exponential random variable with parameter lambda_val.
    """
    return - (1 / lambda_val) * math.log(random.random())

# Test Problem 1
lambda_val = 75
n_samples = 1000
exp_samples = [generate_exponential(lambda_val) for _ in range(n_samples)]
print(f"Generated Mean: {np.mean(exp_samples)}")
print(f"Theoretical Mean: {1 / lambda_val}")
print(f"Generated Variance: {np.var(exp_samples)}")
print(f"Theoretical Variance: {1 / (lambda_val ** 2)}")

# ==============================
# Problem 2: M/M/1 Queue Simulator
# ==============================
class MM1Queue:
    def __init__(self, lambda_val, L, C):
        self.lambda_val = lambda_val  # Arrival rate (packets/second)
        self.L = L  # Average packet length (bits)
        self.C = C  # Link capacity (bits/second)
        self.mu = C / L  # Service rate (packets/second)
        self.rho = (L * lambda_val) / C  # Utilization
        self.queue = []  # Queue to store packets
        self.current_time = 0  # Current simulation time
        self.num_packets = 0  # Number of packets in the system
        self.total_packets_processed = 0  # Total packets processed
        self.total_idle_time = 0  # Total time the server is idle
        self.total_sojourn_time = 0  # Total time packets spend in the system
        self.event_scheduler = []  # Event scheduler (priority queue)
        self.observer_events = []  # Observer events

    def schedule_event(self, event_time, event_type):
        """
        Schedule an event in the event scheduler.
        """
        heapq.heappush(self.event_scheduler, (event_time, event_type))

    def run_simulation(self, simulation_time, alpha):
        """
        Run the M/M/1 queue simulation for a given duration with observer events.
        """
        # Schedule the first arrival event
        self.schedule_event(self.current_time + generate_exponential(self.lambda_val), 'arrival')

        # Schedule observer events
        while self.current_time < simulation_time:
            observer_time = self.current_time + generate_exponential(alpha)
            if observer_time < simulation_time:
                self.schedule_event(observer_time, 'observer')
            else:
                break

        next_departure_time = float('inf')  # No departure initially

        while self.current_time < simulation_time and self.event_scheduler:
            event_time, event_type = heapq.heappop(self.event_scheduler)
            self.current_time = event_time

            if event_type == 'arrival':
                # Arrival event
                if self.num_packets == 0:
                    # Server was idle, now busy
                    next_departure_time = self.current_time + generate_exponential(self.mu)
                    self.schedule_event(next_departure_time, 'departure')
                self.num_packets += 1
                self.total_packets_processed += 1
                self.queue.append(self.current_time)  # Record arrival time
                self.schedule_event(self.current_time + generate_exponential(self.lambda_val), 'arrival')

            elif event_type == 'departure':
                # Departure event
                self.num_packets -= 1
                arrival_time = self.queue.pop(0)
                self.total_sojourn_time += self.current_time - arrival_time
                if self.num_packets > 0:
                    next_departure_time = self.current_time + generate_exponential(self.mu)
                    self.schedule_event(next_departure_time, 'departure')
                else:
                    next_departure_time = float('inf')

            elif event_type == 'observer':
                # Observer event
                self.observer_events.append(self.num_packets)
                if self.num_packets == 0:
                    self.total_idle_time += 1

        # Calculate performance metrics
        avg_packets = np.mean(self.observer_events)
        idle_proportion = self.total_idle_time / len(self.observer_events)
        avg_sojourn_time = self.total_sojourn_time / self.total_packets_processed
        return avg_packets, idle_proportion, avg_sojourn_time

# Test Problem 2 with observer events
lambda_val = 0.8  # Arrival rate (packets/second)
L = 12000  # Average packet length (bits)
C = 1e6  # Link capacity (bits/second)
alpha = 0.8  # Observer event rate (events/second)
simulation_time = 1000  # Simulation duration (seconds)

mm1_queue = MM1Queue(lambda_val, L, C)
avg_packets, idle_proportion, avg_sojourn_time = mm1_queue.run_simulation(simulation_time, alpha)
print(f"Average packets in system: {avg_packets}")
print(f"Proportion of idle time: {idle_proportion}")
print(f"Average sojourn time: {avg_sojourn_time}")

# ==============================
# Problem 3: Simulating M/M/1 Queue Performance
# ==============================
def simulate_mm1_queue_performance(lambda_val, L, C, rho_values, simulation_time):
    """
    Simulate M/M/1 queue for different utilization values and plot results.
    """
    avg_packets_list = []
    idle_proportion_list = []

    for rho in rho_values:
        lambda_val = (rho * C) / L
        mm1_queue = MM1Queue(lambda_val, L, C)
        avg_packets, idle_proportion, _ = mm1_queue.run_simulation(simulation_time)
        avg_packets_list.append(avg_packets)
        idle_proportion_list.append(idle_proportion)

    # Plot E[N] vs rho
    plt.figure()
    plt.plot(rho_values, avg_packets_list, marker='o')
    plt.xlabel("Utilization (ρ)")
    plt.ylabel("Average packets in system (E[N])")
    plt.title("E[N] vs ρ for M/M/1 queue")
    plt.grid(True)
    plt.show()

    # Plot P_idle vs rho
    plt.figure()
    plt.plot(rho_values, idle_proportion_list, marker='o')
    plt.xlabel("Utilization (ρ)")
    plt.ylabel("Proportion of idle time (P_idle)")
    plt.title("P_idle vs ρ for M/M/1 queue")
    plt.grid(True)
    plt.show()

# Test Problem 3
rho_values = np.arange(0.25, 0.96, 0.1)
L = 12000  # Average packet length (bits)
C = 1e6  # Link capacity (bits/second)
# simulate_mm1_queue_performance(lambda_val, L, C, rho_values, 100000)

# ==============================
# Problem 4: High Utilization Simulation (ρ = 1.2)
# ==============================
def simulate_high_utilization(rho, L, C, simulation_time):
    """
    Simulate M/M/1 queue under high utilization (ρ > 1).
    """
    lambda_val = (rho * C) / L  # Calculate arrival rate based on ρ
    mm1_queue = MM1Queue(lambda_val, L, C)
    avg_packets, idle_proportion, avg_sojourn_time = mm1_queue.run_simulation(simulation_time)
    print(f"High Utilization (ρ = {rho}):")
    print(f"Arrival rate (λ): {lambda_val} packets/second")
    print(f"Average packets in system: {avg_packets}")
    print(f"Proportion of idle time: {idle_proportion}")
    print(f"Average sojourn time: {avg_sojourn_time}")

# Test Problem 4
rho = 1.2  # Utilization
L = 12000  # Average packet length (bits)
C = 1e6  # Link capacity (bits/second)
# simulate_high_utilization(rho, L, C, 100000)

# ==============================
# Problem 5: M/M/1/K Queue Simulator
# ==============================
class MM1KQueue(MM1Queue):
    def __init__(self, lambda_val, L, C, K):
        super().__init__(lambda_val, L, C)
        self.K = K  # Buffer size
        self.total_packets_dropped = 0  # Total packets dropped due to buffer full

    def run_simulation(self, simulation_time, alpha):
        """
        Run the M/M/1/K queue simulation for a given duration with observer events.
        """
        # Schedule the first arrival event
        self.schedule_event(self.current_time + generate_exponential(self.lambda_val), 'arrival')

        # Schedule observer events
        while self.current_time < simulation_time:
            observer_time = self.current_time + generate_exponential(alpha)
            if observer_time < simulation_time:
                self.schedule_event(observer_time, 'observer')
            else:
                break

        next_departure_time = float('inf')  # No departure initially

        while self.current_time < simulation_time and self.event_scheduler:
            event_time, event_type = heapq.heappop(self.event_scheduler)
            self.current_time = event_time

            if event_type == 'arrival':
                # Arrival event
                if self.num_packets < self.K:
                    self.num_packets += 1
                    self.total_packets_processed += 1
                    self.queue.append(self.current_time)  # Record arrival time
                    if self.num_packets == 1:
                        # Server was idle, now busy
                        next_departure_time = self.current_time + generate_exponential(self.mu)
                        self.schedule_event(next_departure_time, 'departure')
                else:
                    # Buffer is full, drop the packet
                    self.total_packets_dropped += 1
                self.schedule_event(self.current_time + generate_exponential(self.lambda_val), 'arrival')

            elif event_type == 'departure':
                # Departure event
                self.num_packets -= 1
                arrival_time = self.queue.pop(0)
                self.total_sojourn_time += self.current_time - arrival_time
                if self.num_packets > 0:
                    next_departure_time = self.current_time + generate_exponential(self.mu)
                    self.schedule_event(next_departure_time, 'departure')
                else:
                    next_departure_time = float('inf')

            elif event_type == 'observer':
                # Observer event
                self.observer_events.append(self.num_packets)
                if self.num_packets == 0:
                    self.total_idle_time += 1

        # Calculate performance metrics
        avg_packets = np.mean(self.observer_events)
        idle_proportion = self.total_idle_time / len(self.observer_events)
        avg_sojourn_time = self.total_sojourn_time / self.total_packets_processed
        loss_probability = self.total_packets_dropped / (self.total_packets_processed + self.total_packets_dropped)
        return avg_packets, idle_proportion, avg_sojourn_time, loss_probability

# Test Problem 5
lambda_val = 0.8  # Arrival rate (packets/second)
L = 12000  # Average packet length (bits)
C = 1e6  # Link capacity (bits/second)
K = 10  # Buffer size
mm1k_queue = MM1KQueue(lambda_val, L, C, K)
avg_packets, idle_proportion, avg_sojourn_time, loss_probability = mm1k_queue.run_simulation(100000)
print(f"Average packets in system: {avg_packets}")
print(f"Proportion of idle time: {idle_proportion}")
print(f"Average sojourn time: {avg_sojourn_time}")
print(f"Packet loss probability: {loss_probability}")

# ==============================
# Problem 6: Simulate M/M/1/K Queue for Different Buffer Sizes
# ==============================
def simulate_mm1k_queue(lambda_val, L, C, K_values, simulation_time):
    """
    Simulate M/M/1/K queue for different buffer sizes and plot results.
    """
    # Define ρ values for E[N] and P_loss plots
    rho_values_e_n = np.arange(0.5, 1.6, 0.1)  # For E[N]: 0.5 < ρ < 1.5, step 0.1
    rho_values_p_loss = np.concatenate([
        np.arange(0.4, 2.1, 0.1),  # For P_loss: 0.4 < ρ ≤ 2, step 0.1
        np.arange(2.2, 5.1, 0.2),  # For P_loss: 2 < ρ ≤ 5, step 0.2
        np.arange(5.4, 10.0, 0.4)  # For P_loss: 5 < ρ < 10, step 0.4
    ])

    # Initialize results dictionaries
    results_e_n = {K: {"rho": [], "E[N]": []} for K in K_values}
    results_p_loss = {K: {"rho": [], "P_loss": []} for K in K_values}

    # Simulate for E[N] vs ρ
    for rho in rho_values_e_n:
        lambda_val = (rho * C) / L
        for K in K_values:
            mm1k_queue = MM1KQueue(lambda_val, L, C, K)
            avg_packets, _, _, _ = mm1k_queue.run_simulation(simulation_time)
            results_e_n[K]["rho"].append(rho)
            results_e_n[K]["E[N]"].append(avg_packets)

    # Simulate for P_loss vs ρ
    for rho in rho_values_p_loss:
        lambda_val = (rho * C) / L
        for K in K_values:
            mm1k_queue = MM1KQueue(lambda_val, L, C, K)
            _, _, _, loss_probability = mm1k_queue.run_simulation(simulation_time)
            results_p_loss[K]["rho"].append(rho)
            results_p_loss[K]["P_loss"].append(loss_probability)

    # Plot E[N] vs ρ
    plt.figure()
    for K in K_values:
        plt.plot(results_e_n[K]["rho"], results_e_n[K]["E[N]"], label=f"K={K}")
    # Add K=∞ case (M/M/1 queue)
    rho_values_e_n_inf = np.arange(0.5, 1.5, 0.1)
    e_n_inf = [rho / (1 - rho) if rho < 1 else np.nan for rho in rho_values_e_n_inf]
    plt.plot(rho_values_e_n_inf, e_n_inf, label="K=∞", linestyle="--")
    plt.xlabel("Utilization (ρ)")
    plt.ylabel("Average packets in system (E[N])")
    plt.title("E[N] vs ρ for different buffer sizes")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot P_loss vs ρ
    plt.figure()
    for K in K_values:
        plt.plot(results_p_loss[K]["rho"], results_p_loss[K]["P_loss"], label=f"K={K}")
    plt.xlabel("Utilization (ρ)")
    plt.ylabel("Packet loss probability (P_loss)")
    plt.title("P_loss vs ρ for different buffer sizes")
    plt.legend()
    plt.grid(True)
    plt.show()

# Test Problem 6
K_values = [5, 10, 40]
L = 12000  # Average packet length (bits)
C = 1e6  # Link capacity (bits/second)
simulate_mm1k_queue(lambda_val, L, C, K_values, 1000)
