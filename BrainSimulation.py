import torch
import torch.nn as nn
import numpy as np
import random

class BrainSimulation:
    def __init__(self, total_neurons, complexity, input_size, output_size):
        self.total_neurons = total_neurons
        self.complexity = complexity
        self.neuron_distribution = self.distribute_neurons()
        
        self.components = {
            'neocortex': TransformerModel(self.neuron_distribution['neocortex'], complexity, input_size, output_size),
            'cerebellum': CerebellumModel(self.neuron_distribution['cerebellum'], complexity, input_size, output_size),
            'hippocampus': LSTMWithAttention(self.neuron_distribution['hippocampus'], complexity, input_size, output_size),
            'basal_ganglia': QLearningModel(self.neuron_distribution['basal_ganglia'], complexity, input_size, output_size),
            'thalamus': ThalamusModel(self.neuron_distribution['thalamus'], complexity, input_size, output_size)
        }
        
        self.connection_mask = self.initialize_connection_mask()

    def distribute_neurons(self):
        return {
            'cerebellum': int(self.total_neurons * 0.80),
            'neocortex': int(self.total_neurons * 0.19),
            'hippocampus': int(self.total_neurons * 0.004),
            'basal_ganglia': int(self.total_neurons * 0.005),
            'thalamus': int(self.total_neurons * 0.001)
        }

    def initialize_connection_mask(self):
        return {(from_comp, to_comp): np.random.uniform(0, 20, (self.complexity, self.complexity)) 
                for from_comp in self.components for to_comp in self.components}

    def forward(self, input_data):
        component_outputs = {name: component(input_data) for name, component in self.components.items()}
        final_output = self.apply_connections(component_outputs)
        return final_output

    def apply_connections(self, component_outputs):
        for (from_comp, to_comp), mask in self.connection_mask.items():
            activation = torch.sigmoid(torch.tensor(mask))  # Apply sigmoid activation
            component_outputs[to_comp] += activation * component_outputs[from_comp]
        return sum(component_outputs.values())

    def mutate_mask(self, mutation_rate=0.1, mutation_strength=0.2):
        for mask in self.connection_mask.values():
            mutation = np.random.uniform(-mutation_strength, mutation_strength, mask.shape)
            mask += np.random.random(mask.shape) < mutation_rate * mutation
            np.clip(mask, 0, 20, out=mask)

# ? AI Models 
class TransformerModel(nn.Module):
    def __init__(self, num_neurons, complexity, input_size, output_size):
        super().__init__()
        self.num_layers = complexity
        self.input_size = input_size
        self.output_size = output_size
        neurons_per_layer = max(1, num_neurons // self.num_layers)
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=neurons_per_layer, nhead=1)
            for _ in range(self.num_layers)
        ])
        self.fc_out = nn.Linear(neurons_per_layer, output_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(x)

class CerebellumModel(nn.Module):
    def __init__(self, num_neurons, complexity, input_size, output_size):
        super().__init__()
        self.adaptive_filter = nn.Linear(input_size, num_neurons)
        self.echo_state = nn.RNN(num_neurons, num_neurons, num_layers=complexity, nonlinearity='tanh')
        self.fc_out = nn.Linear(num_neurons, output_size)
        self.output_size = output_size
        self.input_size = input_size
        

    def forward(self, x):
        x = torch.tanh(self.adaptive_filter(x))
        x, _ = self.echo_state(x)
        return self.fc_out(x)

class LSTMWithAttention(nn.Module):
    def __init__(self, num_neurons, complexity, input_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, num_neurons, num_layers=complexity)
        self.attention = nn.MultiheadAttention(num_neurons, num_heads=1)
        self.fc_out = nn.Linear(num_neurons, output_size)
        self.output_size = output_size
        self.input_size = input_size
        

    def forward(self, x):
        x, _ = self.lstm(x)
        x, _ = self.attention(x, x, x)
        return self.fc_out(x)

class QLearningModel(nn.Module):
    def __init__(self, num_neurons, complexity, input_size, output_size):
        super().__init__()
        layers = [nn.Linear(input_size, num_neurons), nn.ReLU()]
        for _ in range(complexity - 1):
            layers.extend([nn.Linear(num_neurons, num_neurons), nn.ReLU()])
        layers.append(nn.Linear(num_neurons, output_size))
        self.q_network = nn.Sequential(*layers)
        self.output_size = output_size
        self.input_size = input_size
        

    def forward(self, x):
        return self.q_network(x)

class ThalamusModel(nn.Module):
    def __init__(self, num_neurons, complexity, input_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, num_neurons, num_layers=complexity)
        self.fc_out = nn.Linear(num_neurons, output_size)
        self.output_size = output_size
        self.input_size = input_size
        

    def forward(self, x):
        x, _ = self.lstm(x)
        return self.fc_out(x)
        
# ? Population Creation
def create_population(population_size, total_neurons, complexity, input_size, output_size):
    return [BrainSimulation(total_neurons, complexity, input_size, output_size) 
            for _ in range(population_size)]

def evaluate_population(population, evaluation_function):
    return [evaluation_function(brain) for brain in population]

def select_parents(population, scores):
    sorted_population = [x for _, x in sorted(zip(scores, population), key=lambda pair: pair[0], reverse=True)]
    return sorted_population[:len(population)//2]

def crossover(parent1, parent2):
    child = BrainSimulation(parent1.total_neurons, parent1.complexity, 
                            next(iter(parent1.components.values())).input_size, 
                            next(iter(parent1.components.values())).output_size)
    
    for (from_comp, to_comp) in parent1.connection_mask.keys():
        child.connection_mask[(from_comp, to_comp)] = (parent1.connection_mask[(from_comp, to_comp)] + parent2.connection_mask[(from_comp, to_comp)]) / 2
    
    return child

def evolve_population(population, evaluation_function, generations=100):
    for generation in range(generations):
        scores = evaluate_population(population, evaluation_function)
        parents = select_parents(population, scores)
        
        new_population = parents.copy()
        while len(new_population) < len(population):
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            child.mutate_mask()
            new_population.append(child)
        
        population = new_population
        
        best_score = max(scores)
        avg_score = sum(scores) / len(scores)
        print(f"Generation {generation}: Best Score = {best_score:.2f}, Average Score = {avg_score:.2f}")
    
    return population

# Example usage
def example_evaluation_function(brain):
    # This is a placeholder evaluation function
    # In a real scenario, you would test the brain on some task and return a score
    return random.random()  # Returns a random score between 0 and 1

if __name__ == "__main__":
    population_size = 10
    total_neurons = 1000
    complexity = 5
    input_size = 10
    output_size = 10

    initial_population = create_population(population_size, total_neurons, complexity, input_size, output_size)
    evolved_population = evolve_population(initial_population, example_evaluation_function)

    best_brain = max(evolved_population, key=example_evaluation_function)
    print("Evolution complete. Best brain found.")

    # Example forward pass with the best brain
    input_data = torch.randn(1, input_size)
    output = best_brain(input_data)
    print(f"Output shape: {output.shape}")