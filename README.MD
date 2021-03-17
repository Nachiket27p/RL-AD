# Reinforcement Learning with Planning to Train Autonomous Driving Agent

## Project Abstract
* The main motivation behind this project is to explore the effectiveness of reinforcement learning for the purpose of autonomous driving.
* Another motivation is the increasing demand for autonomous vehicles for the foreseeable future.
* The main objective is to train an agent to perform autonomous driving in city and highway both, this will required training the agent on both types of environments.
* For the purpose of this implementation a simulation environment such as the one provided by [AirSim] will have to be used for testing purposes.
* There are different methods of performing reinforcement learning, some use as single agent, some use multi agents and currently the objective is to utilize just one agent but with further research perhaps multiple agents might yield a better outcome.
* There are also various methodologies, some only use reinforcement learning, others use a combination of planning and reinforcement learning. For the current standpoint the planning combines with reinforcement leaning has shown more effective autonomous driving capabilities but is also more complicated to implement.
* The full project [proposal]:



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
    * AirSimNH - A small neighbouthood environmnet with moving deers that sometimes run onto the road.
  * Created samll script to run test environment to ensure they run sufficiently.
    * Set testing resolution to 720p for both environments
* 03/07/21
  * Followed [AirSim documentation] to deploy a Microsoft Azure NV6 series virtual machine for training purposes.
  * Also compiled AirSim for windows because the Azure VM is windows.
  * With the main development system chaning to Windows, an additional environment can be added to the list of training environments.
    * City - A large city environment with stationary traffic to navigate, no pedestarians.
  * Pushed compiled AirSim to github.
* 03/15/21
  * Created a modified version of the base model provided
  * Modified to redice amount of image resizing by collecing exact image size for performance reasons (~10%)
  * Explored three algorithms:
    * DQN (Deep Q Network)
    * QRDQN (Quantile Regression DQN)
    * HER (Hindsight Experience Replay)
  * 


....


## Implementation
### Setup Environment
* Obtain AirSim binaries along with the two environments; Blocks, and AirSimNH.
* Follow AirSim instruction and build it
* This specific implementation will utilize Microsoft Azure NV6 series virtual machines to do the training.
* 

....


[proposal]:https://github.com/Nachiket27p/Reinforcement_Learning-Autonomous_Driving/blob/master/Paper/csc720_project_proposal.pdf
[AirSim]:https://github.com/microsoft/AirSim
[AirSim documentation]:https://microsoft.github.io/AirSim/
