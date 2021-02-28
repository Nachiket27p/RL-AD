# Reinforcement Learning with Planning to Train Autonomous Driving Agent

## Project Abstract
* The main motivation behind this project is to explore the effectiveness of reinforcement learning for the purpose of autonomous driving.
* Another motivation is the increasing demand for autonomous vehicles for the foreseeable future.
* The main objective is to train an agent to perform autonomous driving in city and highway both, this will required training the agent on both types of environments.
* For the purpose of this implementation a simulation environment such as the one provided by [AirSim] will have to be used for testing purposes.
* There are different methods of performing reinforcement learning, some use as single agent, some use multi agents and currently the objective is to utilize just one agent but with further research perhaps multiple agents might yield a better outcome.
* There are also various methodologies, some only use reinforcement learning, others use a combination of planning and reinforcement learning. For the current standpoint the planning combines with reinforcement leaning has shown more effective autonomous driving capabilities but is also more complicated to implement.
* The full project proposal: [Proposal]

[Proposal]:https://github.com/Nachiket27p/Reinforcement_Learning-Autonomous_Driving/blob/master/Paper/csc720_project_proposal.pdf
[AirSim]:https://github.com/microsoft/AirSim

## Progress Update
* 02/16/21
  * Created project proposal
* 02/21/21
  * Explored explored various training environments in AirSim.
* 02/22/21
  * Discovered CARLA simulator, another simulation environment.
* 02/28/21
  * Selected two traning environments from AirSim. Decided to continue with AirSim instead of CARLA.
    * Blocks - A open sandbox to perform initial testing
    * AirSimNH - A small neighbouthood environmnet
  * Created samll script to run test environment to ensure they run sufficiently.
    * Set testing resolution to 720p for both environments
* 


....


## Implementation
### Setup Environment
* Obtain AirSim binaries along with the two environments; Blocks, and AirSimNH.
* Follow AirSim instruction and build it
* ...

....