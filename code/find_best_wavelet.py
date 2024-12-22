import numpy as np
import find_filter_coefficients as find_fc
import fitness_function as ff
import find_wavelet_coefficients as fwc

def pso_search_process(X_train, y_train, S, iterations, w, c1, c2, d, eps, levels, coeff_type):
    """
    Perform Particle Swarm Optimization (PSO) to find the best wavelet coefficients for signal processing.
    
    Parameters:
        X_train (array-like): Training data features.
        y_train (array-like): Training data labels.
        S (int): Number of particles in the swarm.
        iterations (int): Number of iterations for the PSO algorithm.
        w (float): Inertia weight.
        c1 (float): Cognitive component coefficient.
        c2 (float): Social component coefficient.
        d (int): Dimensionality of the search space.
        eps (float): Small value for termination tolerance.
        levels (int): Number of decomposition levels for wavelet transform.
        coeff_type (str): Type of wavelet coefficients to use (e.g., 'approximation', 'detail').
    
    Returns:
        tuple: A tuple containing:
            - global_best_position (array-like): The best position found by the swarm.
            - overall_best_training_accuracy (list): List of best training accuracies over iterations.
    """

    # Initialize the position vectors with random values uniformly distributed in [0, 2π)
    positions = np.random.uniform(0, 2 * np.pi, (S, d)) 

    # Initialize the velocity vectors to zero
    velocities = np.zeros((S, d))

    # Initialize the fitness values to zero
    fitness = np.zeros((S))

    # Initialize the personal best positions to the initial positions
    #   personal best_positions  
    #   first 10 numbers in a row -- best positions
    #   11h number in a row       -- respective fitness value
    personal_best_positions = np.zeros((S, d + 1))
    personal_best_positions[:, :-1] = positions.copy()  # Store positions in the first d columns

    # Initialize the global best position to the best position in the swarm
    global_best_position = personal_best_positions[0, :-1].copy()  # Initialize to the first particle's position


    overall_best_training_accuracy = []

    for t in range(iterations):
        
        
        lowpass_filter_bank =  find_fc.generate_filter_bank(positions)[0]
        highpass_filter_bank = find_fc.generate_filter_bank(positions)[1]

        # Prepare the dataset for the fitness function evaluation
        X_train_svm =fwc.update_wavelet_coefficients(X_train, lowpass_filter_bank, highpass_filter_bank, levels, coeff_type)
        
        fitness =ff.fitness_function_CV(X_train_svm, y_train)

        #Update Global and Personal best 
        for i in range(S):
            if fitness[i]> personal_best_positions[i,-1]:
                personal_best_positions[i,:-1] = positions[i,:]
                personal_best_positions[i,-1] = fitness[i]

        global_best_position = personal_best_positions[np.argmax(personal_best_positions[:, -1]), :-1]  # Exclude fitness from best position
        overall_best_training_accuracy.append(max(personal_best_positions[:, -1]))
        
        # Random weights from uniform distribution [0,1]
        r1, r2 = np.random.uniform(0,1), np.random.uniform(0,1)
        
        for i in range(S):

            for j in range(d):

                # Calculate velocity based on inertia, cognitive, and social components
                velocities[i,j] = (
                    w * velocities[i,j] +
                    c1 * r1 * (personal_best_positions[i,j] - positions[i,j]) +
                    c2 * r2 * (global_best_position[j] - positions[i,j])
                )

                # Update position based on the new velocity
                positions[i,j] += velocities[i,j]
        print("iteration", t, "completed")
    return global_best_position, overall_best_training_accuracy