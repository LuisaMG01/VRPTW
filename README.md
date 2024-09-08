# Vehicle Routing Problem with Time Windows (VRPTW) - Implementation


## Problem Statement

The Vehicle Routing Problem with Time Windows (VRPTW) is a variation of the classical Vehicle Routing Problem (VRP). In the VRPTW, there are a set of customers located at different points on a map, each requiring a certain demand to be met. These customers have a specific time window during which the delivery must occur.

The problem involves a fleet of vehicles starting at a depot, delivering goods to the customers while adhering to capacity constraints and the customers' time windows. The goal is to minimize amount of vehicles used while ensuring that all deliveries are made within the given time constraints.

## Key Constraints:

- Each vehicle has a limited capacity, which cannot be exceeded by the demands of the customers.
- Each customer must be visited within a specific time window.
- The goal is to find multiple routes that respect the time and capacity constraints while minimizing the amount of vehicles.

  ## Summary of Solutions:


  ### Constructive Heuristic:
  This is the baseline solution for the VRPTW. It builds a solution by incrementally assigning customers to vehicles based on a set of feasibility conditions:
- The vehicle must not exceed its capacity.
- The arrival time at the customer’s location must be within the allowed time window.
- Once a customer is assigned to a vehicle, it is removed from the unassigned list. A greedy approach is used to pick the next customer based on proximity to the current customer.
The solution terminates when all customers are assigned or when no feasible assignments can be made.

  ### GRASP Alpha Heuristic:
  The Greedy Randomized Adaptive Search Procedure (GRASP) extends the constructive heuristic by adding randomness to the customer selection process.
A Restricted Candidate List (RCL) is created, containing the best candidate customers based on distance, but a random element is introduced by selecting customers randomly from this list.
This randomness helps explore different solutions across multiple iterations, allowing the algorithm to escape local optima.
The solution is refined using local search procedures to improve the quality of the route.


  ### GRASP Cardinality Heuristic:
  Once a solution is constructed (either through the baseline or GRASP), a local search is applied to improve the routes.
The Two-Opt algorithm is used to optimize the routes by reversing sections of the route and checking whether this leads to an improved total travel distance.
The Two-Opt algorithm works iteratively, trying to improve each route by reducing the total travel distance for each vehicle.

## Project Structure

- instances: This folder contains the instances to run the project:
- analysis.xlsx: This excel file contains the analysis between the algorithms.
- constructive.py: This file contains the constructive heuristic algorithm.
- grasp_alpha.py: This file contains the GRASP algorithm based on alpha.
- grasp_cardinality.py: This file contains the GRASP algorithm based in cardinality.
- lower_bound.py: This file calculates the lower bound for all instances and is used to calculate the GAP into the analysis.xlsx.


## Usage 


### Prerequisites:
- Python 3.10 or higher
- Required Python libraries:
  - openpyxl
  - math
  - time
  - random
 
To install the required libraries create a virtual enviroment and install the ```requirements.txt``` file:

```
python -m venv venv
```

```
pip install -r requirements.txt
```

To run the python files, use the command:

```
python file_name.py
```
After running the file, an excel file with the results will be created.

**NOTE:**
The url with the presentation of the project is: ```https://www.canva.com/design/DAGQG8Czkro/uOvZ-Ksjh1sIPglaQCA2LQ/view?utm_content=DAGQG8Czkro&utm_campaign=designshare&utm_medium=link&utm_source=editor```


