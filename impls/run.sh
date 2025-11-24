############### State Based

## Point Maze

# Large

# ipython main.py -- --env_name=pointmaze-large-navigate-v0 --eval_episodes=25 --agent=agents/dhp_buff.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --run_group=DHP --exp_name=pntL_dhpbuffD8_rchM2Dist2_actTrgtVal

# Giant

ipython main.py -- --env_name=pointmaze-giant-navigate-v0 --eval_episodes=25 --agent=agents/dhp_buff.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --run_group=DHP --exp_name=pntG_dhpbuffD8_rchM2Dist2_actTrgtVal


## Ant Maze

# Large

# ipython main.py -- --env_name=antmaze-large-navigate-v0 --eval_episodes=50 --agent=agents/dhp_2value.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --run_group=DHP --exp_name=antL_dhp2_lowUnifData2

# ipython main.py -- --env_name=antmaze-large-navigate-v0 --eval_episodes=25 --agent=agents/dhp_buff.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --run_group=DHP --exp_name=antL_dhpbuffD5_rchM1P5Dist2By4_2

# ipython main.py -- --env_name=antmaze-large-navigate-v0 --eval_episodes=25 --agent=agents/dhp.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --run_group=DHP --exp_name=antL_dhp4_hpDep1_hPolValStateEnc_8x8

# Giant

# ipython main.py -- --env_name=antmaze-giant-navigate-v0 --eval_episodes=25 --agent=agents/dhp_buff.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --run_group=DHP --exp_name=antG_dhpbuffD8_rchM2Dist2


## Ant Maze Explore

# ipython main.py -- --env_name=antmaze-large-explore-v0 --eval_episodes=25 --agent=agents/dhp_buff.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --run_group=DHP --exp_name=antLExp_dhpbuffD5_rchM1P5Dist2


## Humanoid Maze

# Large

# ipython main.py -- --env_name=humanoidmaze-large-navigate-v0 --eval_episodes=25 --agent=agents/dhp_buff.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --run_group=DHP --exp_name=humL_dhpbuffD8_rchM2Dist2_3 --agent.subgoal_steps=100

# ipython main.py -- --env_name=humanoidmaze-large-navigate-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --run_group=HIQL --exp_name=humL_hiql_100steps --agent.subgoal_steps=100

# Giant

# ipython main.py -- --env_name=humanoidmaze-giant-navigate-v0 --eval_episodes=25 --agent=agents/dhp_buff.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --run_group=DHP --exp_name=humG_dhpbuffD8_rchM1P5Dist2 --agent.subgoal_steps=100


# AntSoccer

# ipython main.py -- --env_name=antsoccer-arena-navigate-v0 --eval_episodes=25 --agent=agents/dhp_buff.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --run_group=DHP --exp_name=antSocA_dhpbuffD8_rchM1P5Dist2



## Puzzle

# 3x3

# ipython main.py -- --env_name=puzzle-3x3-noisy-v0 --eval_episodes=25 --agent=agents/dhp_buff.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --run_group=DHP --exp_name=puzNoi33_dhpbuffD8_rchM2Dist2 --agent.subgoal_steps=10


## Puzzle

# 3x3

# ipython main.py -- --env_name=puzzle-3x3-play-v0 --eval_episodes=25 --agent=agents/dhp_buff.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --run_group=DHP --exp_name=puz33_dhpbuffD8_rchM2Dist2 --agent.subgoal_steps=10



############### VISUAL

# Ant

# ipython main.py -- --env_name=visual-antmaze-large-navigate-v0 --eval_episodes=25 --agent=agents/dhp_buff.py --agent.batch_size=256 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 --exp_name=VantL_dhpbuffD8_rchM2Dist2

# Humanoid

# ipython main.py -- --env_name=visual-humanoidmaze-large-navigate-v0 --eval_episodes=25 --agent=agents/dhp_buff.py --agent.batch_size=256 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 --agent.subgoal_steps=100 --exp_name=VhumL_dhpbuffD8_rchM2Dist2





############### DEBUG

# State Based

# ipython main.py -- --env_name=antmaze-large-navigate-v0 --eval_episodes=1 --agent=agents/dhp_buff.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --debug=True --exp_name=dhpbuff_hpactor_normLowActVal

# ipython main.py -- --env_name=puzzle-3x3-play-v0 --eval_episodes=1 --agent=agents/dhp.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --debug=True

# Image based

# ipython main.py -- --env_name=visual-antmaze-medium-navigate-v0 --eval_episodes=1 --agent=agents/dhp_buff.py --agent.batch_size=256 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 --debug=True
