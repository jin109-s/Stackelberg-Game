# Stackelberg-Game
## Background
**Mobile Edge Computing (MEC)** emerges as a new computing paradigm that can leverage the computing capabilities of both local devices and edge servers simultaneously. In MEC,**Edge Pricing** and **Resource Allocation (EPRA)** are two critical issues. Edge servers generate profits by selling **computing services** to users. To maximize revenue, they need to determine an appropriate price for each user and decide the amount of resources allocated to each user. However, existing approaches fail to consider the impact of users' task allocation strategies on edge revenue. In fact, EPRA affects users' **task offloading decisions**, as users aim to minimize their total costs. Conversely, users' decisions also influence the edge's revenue. Therefore, the interaction between mobile users and edge servers needs to be carefully considered to maximize the benefits of both parties.
The interaction between the two parties is modeled as a **Stackelberg game**. Given an EPRA strategy, a near-optimal task allocation strategy for each user is derived based on **the greedy algorithm UTA-G** to minimize the total cost. Then, using the backward induction method, two pricing and resource allocation schemes with different granularities—**User-Granularity EPRA (EPRA-U)** and **Task-Granularity EPRA (EPRA-T)**—are proposed to bring higher revenue to the edge.  

## Code Directory Structure  

EPRA-Stackelberg-Replication/  
│  
├── models/  
│   ├── **task.py（defines the data model for tasks）**  
│   ├── **user_device.py（defines the data model for user devices）**  
│   ├── **edge_server.py（defines the data type for edge servers）**  
│   └── **cloud_server.py（defines the data model for cloud servers）**  
│  
├── algorithms/  
│   ├── **epra_u.py（implements the EPRA-U algorithm）**  
│   ├── **epra_t.py（implements the EPRA-T algorithm）**  
│   ├── **uta_g.py（implements the UTA-G algorithm）**  
│   └── **baseline.py（implements four baseline algorithms）**  
│  
├── utils/  
│   ├── **cost_functions.py（implements various cost calculation functions）**  
│   ├── **plot_utils.py（provides plotting tools for generating line charts of experimental results）**  
│   └── **network_params.py（defines the system's network parameters and physical constants）**  
│  
├── experiments/  
│   ├── **exp1_tasks.py（implements the experiment on the impact of task quantity on edge revenue）**  
│   ├── **exp2_local_resource.py（implements the experiment on the impact of local resource coefficients on edge revenue）**  
│   ├── **exp3_edge_resource.py（implements the experiment on the impact of edge system coefficients on edge revenue）**  
│   ├── **exp4_input_scale.py（implements the experiment on the impact of input data size on edge revenue）**  
│   └── **helper_generate.py（provides auxiliary generation functions required for experiments）**  
│  
├── **main.py（the program entry file, used to call and execute all experiments）**  

## Running Instructions
1.Clone the code to your local computer;  
2.Install the project dependencies (numpy, matplotlib);  
3.Execute the command `python main.py` to run all experiments.  
After running the algorithms, data related to edge pricing and resource allocation, as well as visual charts, will be generated—this facilitates comparing algorithm performance under different parameters.
