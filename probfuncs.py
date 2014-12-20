from itertools import accumulate
import random
import bisect
from collections import Sequence

def normalize(dist):
    min_val = min(dist) 
    if min_val < 0:
        dist = [x - min_val for x in dist]
        #dist -= min_val 
    normalizer = sum(dist)
    return [x/normalizer for x in dist]

def cumulative(dist):
    return list(accumulate(dist))

def cum_dist_choice(dist):
    val = random.random()
    for i in range(len(dist)):
        if val < dist[i]:
            return i

# http://stackoverflow.com/a/13052108/2224317
def weighted_sample(population, weights, k):
    return random.sample(WeightedPopulation(population, weights), k)

class WeightedPopulation(Sequence):
    def __init__(self, population, weights):
        assert len(population) == len(weights) > 0
        self.population = population
        self.cumweights = cumulative(weights)
    def __len__(self):
        return self.cumweights[-1]
    def __getitem__(self, i):
        if not 0 <= i < len(self):
            raise IndexError(i)
        return self.population[bisect.bisect(self.cumweights, i)]

def increment(counter, value, amount=1):
    if value in counter:
        counter[value] += amount
    else:
        counter[value] = amount

def is_dist(distList):
    return abs(sum(distList) - 1) < .001
