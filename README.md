# OfflineRL
Implementing offline RL algorithm - [IQL](https://arxiv.org/pdf/2110.06169). Heavily based on [link_1](https://github.com/ikostrikov/implicit_q_learning/tree/master) and [link_2](https://github.com/corl-team/CORL/blob/main/algorithms/offline/iql.py).

Installation:
This version was implemented for arm mac - strongly do NOT recommend to do so...
```
git clone https://github.com/sacr1f1ce/OfflineRL.git
cd OfflineRL
bash setup/setup.sh
```
At this point there maybe several problems with different libraries being in the wrong folders. Manually changing the paths to the correct ones is the best solution.

To run training - `python3 train.py`