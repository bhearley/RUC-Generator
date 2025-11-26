# Test Optimizing Microstructures with Random Seed

# Import modules
from baybe import Campaign
from baybe.recommenders import (
    TwoPhaseMetaRecommender,
    NaiveHybridSpaceRecommender,
    )
from baybe.parameters import (
    NumericalContinuousParameter,
    )
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from baybe.objectives import SingleTargetObjective
import numpy as np
import pandas as pd
import random

# Import Functions
from Random.RandomSBD import RandomSBD
from Random.RandomCharacterization import RandomCharacterization


# Define Input Structure
file_path = r"C:\Users\bhearley\Downloads\random_example.csv"
mask = np.loadtxt(file_path, delimiter=",", dtype=int)

# Define Run Settings
batch_size = 10
patience = 50
max_iter = 500
nbins = 10

# Define Constants
W = 100
H = 100
N_fibers = 30
damping = 0.5
k = 5000
dt = 0.01
steps = 5000
VF = np.sum(mask == 1) / (len(mask) * len(mask[0]))

# Define Parameters
params = [
    NumericalContinuousParameter(
        name="seed",         
        bounds=(1., 1.e6)       
    )]

# Initialize data output
mask_out = {}

# Create the Campaign
space = SearchSpace.from_product(parameters = params)
obj = SingleTargetObjective(target=NumericalTarget(name='Target',minimize = True,))
hybrid_recommender = TwoPhaseMetaRecommender(recommender=NaiveHybridSpaceRecommender())
camp = Campaign(
    searchspace=space,
    objective=obj,
    recommender=hybrid_recommender,
)

# Perform initial characterization
mean_vf_true, iqr_vf_true, bin_centers_true, pdf_true = RandomCharacterization(mask, nbins = nbins)

# Create Initial Data
df = pd.DataFrame()
params_init = {}
for i in range(batch_size):
    for j in range(len(params)):
        if i == 0:
            params_init[params[j].name] = []
        try:
            params_init[params[j].name].append(random.uniform(params[j].bounds.lower,params[j].bounds.upper))
        except:
            params_init[params[j].name].append(random.choice(list(params[j].values)))

for key in params_init.keys():
    df[key] = params_init[key]

# Evaluate the objective function
q = []
for i in df.index:
    print(f'Initializing Microstructure {i+1}/{batch_size}')

    # Get Inputs
    seed = df['seed'][i]
    
    # Generate the mask
    data = RandomSBD(W, H, N_fibers, VF, damping, k, dt, steps, min_gap = 1, n_gen = 1, periodic = True, seed = seed)
    mask_i = data[0][1]

    # Characterize the mask
    mean_vf_i, iqr_vf_i, bin_centers_i, pdf_i = RandomCharacterization(mask_i, nbins=nbins)

    # Calculate mean squared error
    error = np.sqrt(np.sum((pdf_i - pdf_true)**2))

    # Update measurement
    q.append(error)

    # Update data structure
    mask_out[seed] = mask_i

# Add results to campaign
df['Target'] = q
camp.add_measurements(df)

# Initialize Bayesian Optimization
best_ovr = []
best_batch = []
patience_ct = 0

current_batch = camp.measurements['BatchNr'].max()
batch_df = camp.measurements[camp.measurements['BatchNr'] == current_batch]
best_batch.append(batch_df['Target'][batch_df['Target'].idxmin()])
best_ovr.append(camp.measurements['Target'][camp.measurements['Target'].idxmin()])
best_prev = best_ovr[-1]

# Run Bayesian Optimization
for i in range(max_iter):
    print(f'Bayesian Optimization Iteration {i+1}/{max_iter}')

    # Create the data frame with BO recommendations
    df_bo = camp.recommend(batch_size=batch_size)

    # Evaluate recommendations
    q_bo = []
    for j in df_bo.index:
        print(f'    Bayesian Recommendation {j+1}/{batch_size}')

        # Get Inputs
        seed = df_bo['seed'][j]
        
        # Generate the mask
        data = RandomSBD(W, H, N_fibers, VF, damping, k, dt, steps, min_gap = 1, n_gen = 1, periodic = True, seed = seed)
        mask_j = data[0][1]

        # Characterize the mask
        mean_vf_j, iqr_vf_j, bin_centers_j, pdf_j = RandomCharacterization(mask_j, nbins=nbins)

        # Calculate mean squared error
        error = np.sqrt(np.sum((pdf_j - pdf_true)**2))

        # Update measurement
        q_bo.append(error)

        # Update data structure
        mask_out[seed] = mask_j

    # Add simulations to campaign
    df_bo['Target']=q_bo
    camp.add_measurements(df_bo)

    # Get Information
    current_batch = camp.measurements['BatchNr'].max()
    batch_df = camp.measurements[camp.measurements['BatchNr'] == current_batch]
    best_batch.append(batch_df['Target'][batch_df['Target'].idxmin()])
    best_ovr.append(camp.measurements['Target'][camp.measurements['Target'].idxmin()])

    # Check patience
    if best_ovr[-1] == best_prev:
        patience_ct = patience_ct + 1
        if patience_ct == patience:
            break
    else:
        patience_ct = 1
        best_prev = best_ovr[-1]

# Save results to csv
pd.DataFrame(best_batch).to_csv("best_in_batch.csv", index=False)
pd.DataFrame(best_ovr).to_csv("best_overall.csv", index=False)

# Get best microstructure
best_idx = camp.measurements['Target'].idxmin()
best_seed = camp.measurements.loc[best_idx, 'seed'] 
best_mask = mask_out[best_seed]
pd.DataFrame(best_mask).to_csv("best_mask.csv", index=False)