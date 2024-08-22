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

The lastest run [results](https://api.wandb.ai/links/darkdestiny/pm4p7zp4). Evaluation_return_distances was added to measure how close the actor gets to the destintation point since the scores are binary and do not reflect progress. Somehow there appears to be a problem in learning the correct behavior. By comparing the loss with other implementations I noticed that my value loss appears to be significantly higher (~ x10). So the bug is probably somwhere in here, I might have messed up some constatnts. Overall, I tried to use the original implementation hyperparameters wherever it is possible. If I am unable to find any bugs with the code, I'll try to check the validity by setting the beta (inverse temperature) parameter very low. Is it stated in the original article that in this case "objective behaves similarly to behavioral cloning" and BC achieves non-zero score on antmaze-umaze-v0.
