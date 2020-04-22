# Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures

This repository contains an implementation of "Importance Weighted Actor-Learner
Architectures", along with a *dynamic batching* module. This is not an
officially supported Google product.

For a detailed description of the architecture please read [our paper][arxiv].
Please cite the paper if you use the code from this repository in your work.

### Bibtex

```
@inproceedings{impala2018,
  title={IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures},
  author={Espeholt, Lasse and Soyer, Hubert and Munos, Remi and Simonyan, Karen and Mnih, Volodymir and Ward, Tom and Doron, Yotam and Firoiu, Vlad and Harley, Tim and Dunning, Iain and others},
  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},
  year={2018}
}
```
## Setup instructions

- Build conda environment
```sh
# make sure you are under this folder
conda env create -f environment.yml 
conda activate scalable_agent
```

- Download deepmind_lab-1.0-py3-none-any.whl from https://drive.google.com/file/d/10SjkK7AcegkWBgx31-fbKTNRgNLd0wax/view?usp=sharing
```sh
pip install deepmind_lab-1.0-py3-none-any.whl
```

## Running the Code

### Single Machine Training on a Single Level

We support three environments: Atari, Vizdoom, DMLab-30.

```sh
# Atari (BreakoutNoFrameskip-v4)
python -m experiment --level_name=atari_breakout --num_actors=4 --logdir=logdir/atari_4_1
```
```sh
# Vizdoom (doom_battle)
python -m experiment --level_name=doom_benchmark --num_actors=4 --logdir=logdir/doom_4_1
```
```sh
# DMLab (rooms_collect_good_objects_train)
python -m experiment --level_name=rooms_collect_good_objects_train --num_actors=4 --logdir=logdir/dmlab_4_1
```

### Distributed Training on DMLab-30

Training on the full [DMLab-30][dmlab30]. Across 10 runs with different seeds
but identical hyperparameters, we observed between 45 and 50 capped human
normalized training score with different seeds (`--seed=[seed]`). Test scores
are usually an absolute of ~2% lower.

#### Learner

```sh
python experiment.py --job_name=learner --task=0 --num_actors=150 \
    --level_name=dmlab30 --batch_size=32 --entropy_cost=0.0033391318945337044 \
    --learning_rate=0.00031866995608948655 \
    --total_environment_frames=10000000000 --reward_clipping=soft_asymmetric
```

#### Actor(s)

```sh
for i in $(seq 0 149); do
  python experiment.py --job_name=actor --task=$i \
      --num_actors=150 --level_name=dmlab30 --dataset_path=[...] &
done;
wait
```

#### Test Score

```sh
python experiment.py --mode=test --level_name=dmlab30 --dataset_path=[...] \
    --test_num_episodes=10
```

[arxiv]: https://arxiv.org/abs/1802.01561
[deepmind_lab]: https://github.com/deepmind/lab
[sonnet]: https://github.com/deepmind/sonnet
[learning_nav]: https://arxiv.org/abs/1804.00168
[generate_images]: https://deepmind.com/blog/learning-to-generate-images/
[tensorflow]: https://github.com/tensorflow/tensorflow
[dockerfile]: Dockerfile
[dmlab30]: https://github.com/deepmind/lab/tree/master/game_scripts/levels/contributed/dmlab30
