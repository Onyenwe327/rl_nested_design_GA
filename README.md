# Dependencies
* Create conda environment by
```
conda create -n "rl-design" python=3.10
```
install pytorch
```
pip install stable-baselines3[extra] &&
pip install deap gymnasium mujoco
pip install brax
pip install mujoco-mjx
pip install mediapy
pip install -U 'jax[cuda12]'
```