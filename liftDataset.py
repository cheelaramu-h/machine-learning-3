def lift(x):
    d = len(x)
    lifted = []
    
    # Add original coordinates
    lifted.extend(x)
    
    # Add coordinate combinations
    for i in range(d):
        for j in range(i, d):
            lifted.append(x[i] * x[j])
    
    return lifted




def liftDataset(X):
    n, d = X.shape
    lifted_X = []
    
    # Lift each row of the dataset
    for row in X:
        lifted_row = lift(row)
        lifted_X.append(lifted_row)
    
    # Convert to numpy array
    lifted_X = np.array(lifted_X)
    
    return lifted_X

