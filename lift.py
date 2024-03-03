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




