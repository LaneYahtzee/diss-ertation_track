from bofire.data_models.features.api import ContinuousInput, DiscreteInput, ContinuousOutput
from bofire.data_models.objectives.api import MaximizeObjective
from bofire.data_models.domain.api import Inputs, Outputs, Domain
import bofire.strategies.api as strategies
from bofire.data_models.surrogates.api import BotorchSurrogates, SingleTaskGPSurrogate
from bofire.data_models.kernels.api import ScaleKernel, RBFKernel
from bofire.data_models.strategies.api import MoboStrategy, QparegoStrategy
from bofire.data_models.acquisition_functions.api import qLogNEHVI
from bofire.utils.multiobjective import compute_hypervolume
import pandas as pd

input_samples = r"C:\Users\Lane\Downloads\MOBO PDMS Round 3\input_samples.csv"

# Define inputs as x. These can be Continuous, Discrete, or Categorical
x1 = ContinuousInput(key="Base", bounds=(5,35))         #Polymer Base
x2 = ContinuousInput(key="Cure", bounds=(1,5))          #Curing Agent
x3 = DiscreteInput(key="Thickness", values=[1,1.5,2])   #Thickness

# Define outputs and optimization goals as y. Objectives must be floats and can be Max or Min. They can also be weighted. 
objective1 = MaximizeObjective( 
    w=1.0, 
    bounds= [0.0,5.0],
)
y1 = ContinuousOutput(key="Complex Modulus", objective=objective1) #Average Complex Modulus from Frequency Sweeping

objective2 = MaximizeObjective(
    w=1.0
)
y2 = ContinuousOutput(key="Slope", objective=objective2) #Slope of resulting line from Compression Testing

input_features = Inputs(features = [x1, x2, x3])
output_features = Outputs(features=[y1, y2])

#Store all information into the Domain() func. to prepare for optimization. 
domain = Domain(
    inputs=input_features, 
    outputs=output_features
    )

# Function that allows for the generation of a uniform sampling of the inputs. I.e. Warmstart Variables.
#WarmStart = input_features.sample(n=10, method=SamplingMethodEnum.LHS)
#print(WarmStart)

# Generating toy data. 
experiments = pd.read_csv(input_samples, header=0)

# def hypervolume(domain: Domain, experiments: pd.DataFrame) -> float:
#     return compute_hypervolume(domain, experiments)

# MOBO approach using qLogNEHVI, two RBFKernels (one per objective), and returning 5 recommendations. 
data_model = MoboStrategy(domain=domain, acquisition_function= qLogNEHVI(),
    surrogate_specs=BotorchSurrogates(surrogates=[
            SingleTaskGPSurrogate(
                inputs=domain.inputs,
                outputs=Outputs(features=[domain.outputs[0]]), 
                kernel=ScaleKernel(base_kernel=RBFKernel(ard=True))),
            SingleTaskGPSurrogate(
                inputs=domain.inputs,
                outputs=Outputs(features=[domain.outputs[1]]),
               kernel=ScaleKernel(base_kernel=RBFKernel(ard=True))),
    ]))
recommender = strategies.map(data_model=data_model)
recommender.tell(experiments=experiments)
candidates = recommender.ask(candidate_count=5)
print(candidates)