# ALVES DE BARROS Marine
# M2 Bioinformatics - University of Bordeaux
# Genetic algorithm to select the best set of 8 features from the scikit learn breast cancer dataset
# Very inspired by "Feature Reduction from Breast Cancer Dataset with Genetic Algorithm" by Sumit Paul.

# This algorithm does not provide the same result each time due to the fact that we are not looking
# for a 100% accuracy. Various samples of features can produce a high accuracy.

# This algorithm is simplified by the fact that the crossovers always happen in the middle of the genomes,
# There are always 4 bits with a value of 1 on each half 
# The mutations always happen on the first half and always keep the "4 bits with a value of 1" rule.


from sklearn.datasets import load_breast_cancer
import numpy as np
import random
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def generate_population(pop_size,genome_size):
    '''
    Generates random genomes for the algorithm
    A genome is a sample (list) of bits of size (genome_size), 
    each bit corresponding to a feature
    A population is a list of genomes of size (pop_size)
    There are 4 bits with a value of 1 on each half of the genome.
    Returns a population
    '''
    population = np.zeros([pop_size,genome_size], dtype=int)
    for i in range(pop_size):
        for j in range(4):
            position1 = random.randint(0,genome_size/2)
            position2 = random.randint(genome_size/2,genome_size-1)
            while (population[i][position1] == 1):
                position1 = random.randint(0,genome_size/2)
            while (population[i][position2] == 1):
                position2 = random.randint(genome_size/2,genome_size-1)
            population[i][position1] = 1
            population[i][position2] = 1
    return population


def getCorrespondingFeatures(genome):
    '''
    For a genome (string of bits) : gets the features corresponding to a bit value of 1
    '''
    features = []
    for i in range( len(genome)):
        if genome[i] == 1:
            features.append(i)
    return features


def fitness(population, number_of_features, X, Y):
    '''
    Computes the fitness with a SVM algorithm : 
    Each genome (sample of features) is used to run a SVM algorithm, 
    the accuracy of the SVM corresponds to the fitness of the SVM for the genome used to train and test the algorithm.
    Returns an array of the accuracies.
    The better the accuracy is, the better the genome is useful to classify the data.
    '''
    genome_accuracies = []
    for i in range(len(population)):
        features = getCorrespondingFeatures(population[i])
        X_i = X[:, features]
        Y_i = Y
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_i, Y_i, test_size=0.2)
        model = SVC(kernel='linear', random_state=0)
        hist = model.fit(X_train, Y_train)
        Y_predicted = hist.predict(X_test)
        accuracy = accuracy_score(Y_predicted, Y_test)
        genome_accuracies.append(accuracy)
    return genome_accuracies


def selection(population, genome_accuracies):
    '''
    Sorts the fitnesses (accuracies) to select the best 4 of the list
    then gets the corresponding genome in the population.
    Returns the 4 best genomes
    '''
    best_genomes = []
    features_indexes = []

    # Sorting the accuracies
    genome_accuracies_sorted = genome_accuracies.copy()
    genome_accuracies_sorted = genome_accuracies_sorted
    genome_accuracies_sorted.sort(reverse=True)

    # Selecting the best 4
    best_4_accuracies = genome_accuracies_sorted
    best_4_accuracies = best_4_accuracies[0:4]

    # Getting the corresponding genomes
    for i in range(len(best_4_accuracies)):
        features_indexes.append(genome_accuracies.index(best_4_accuracies[i]))
    for i in (features_indexes):
        best_genomes.append(population[i])
    return best_genomes


def crossover(best_genomes):
    '''
    Creates 4 new genomes, crossing the first one with the second one, 
    and the third one with the fourth one.
    Returns these genomes.
    '''
    new_genomes = np.zeros([len(best_genomes), len(best_genomes[0])])

    for i in range(0, len(best_genomes)):
        for j in range(0, len(best_genomes[0])):

            # Doing the crossover in the middle of the genomes
            if j < (len(best_genomes[0])/2):
                new_genomes[i][j] = best_genomes[i][j]
            else:
                # Creating the 2 first new genomes with the beginning of the
                # first and third, and the end of the second and fourth
                if i == 0 and i == 2:
                    feature_index = 0
                    while(feature_index < len(best_genomes[0])):
                        if best_genomes[i+1][feature_index] not in new_genomes[i]:
                            new_genomes[i][j] = best_genomes[i +
                                                             1][feature_index]
                            break
                        feature_index += 1
                # Creating the 2 others, beginning of the
                # second and fourth, and the end of the first and third
                else:
                    feature_index = 0
                    while(feature_index < len(best_genomes[1])):
                        if best_genomes[i-1][feature_index] not in new_genomes[i]:
                            new_genomes[i][j] = best_genomes[i -
                                                             1][feature_index]
                            break
                        feature_index += 1
    return new_genomes


def mutation(crossed_genomes):
    '''
    For each genome, selects a random bit with a value of 0 and turns it to 1
    Does the same for a bit with a value of 1, turns it to 0
    To simplify the code, it chooses randomly on the first half of the genome (to keep 4 bits with a value of 1 on each side.)
    Returns the genomes mutated.
    '''
    for i in range(len(crossed_genomes)):
        old_bite = 1
        position = 0
        while (old_bite != 0):
            position = random.randint(0, 14)
            old_bite = crossed_genomes[i][position]
        crossed_genomes[i][position] = 1

        old_bite = 0
        position = 0
        while (old_bite != 1):
            position = random.randint(0, 14)
            old_bite = crossed_genomes[i][position]
        crossed_genomes[i][position] = 0
    return crossed_genomes


def run_algorithm(data):
    '''
    Runs the algorithm until finding the best 8 features of the dataset
    '''

    # Loading feature names, feature values and labels
    features = data.feature_names
    X = data.data
    Y = data.target

    # Setting parameters
    number_of_features = 8
    best_fitness_so_far = 0
    until_best = []

    iteration_number = 1

    # Generating the random population
    generation = generate_population(8, 30)

    # ----- Running the algorithm ----- #
    run_algorithm = True
    while(run_algorithm):
        print("Iteration: ", iteration_number)

        # Computing the fitnesses of the generation
        # and keeping the best in variable "best_fitness_so_far" and "until_best"
        fitnesses = fitness(generation, number_of_features, X, Y) 
        best_fitness_so_far = max(fitnesses) 
        until_best.append(best_fitness_so_far) 

        # If the best fitness of this generation isn't better
        # than the maximum found, end of the while loop.
        if(best_fitness_so_far < max(until_best)):
            run_algorithm = False
            break
        
        #Selecting the best genomes of the generation
        best_genomes = selection(generation, fitnesses) 
        best_features = []
        best_features.append(getCorrespondingFeatures(best_genomes[0]))
        genomes_crossed = crossover(best_genomes)
        mutated_genomes = mutation(genomes_crossed)
        half_size_of_generation = int(len(generation[0])/2)
        for i in range(int((len(generation)/2))):
            generation[i] = best_genomes[i]

        for j in range(len(mutated_genomes)):
            i += 1
            generation[i] = mutated_genomes[j]
        iteration_number += 1

    print("Best Features : ", best_features)
    for i in range (len(best_features)):
        print (features[best_features[i]])
    print("Accuracy : ", until_best[-1])


data = load_breast_cancer()
run_algorithm(data)
