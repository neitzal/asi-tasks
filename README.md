# Sequential prediction tasks for Adaptive Skip Intervals
This repository includes visual prediction tasks for the paper
[Adaptive Skip Intervals: Temporal Abstraction for Recurrent Dynamical Models](https://arxiv.org/abs/1808.04768)

```
@article{neitz2018adaptive,
  title={Adaptive Skip Intervals: Temporal Abstraction for Recurrent Dynamical Models},
  author={Neitz, Alexander and Parascandolo, Giambattista and Bauer, Stefan and Sch{\"o}lkopf, Bernhard},
  journal={arXiv preprint arXiv:1808.04768},
  year={2018}
}
```

See repository [adaptive-skip-intervals](https://github.com/neitzal/adaptive-skip-intervals) for an implementation of the ASI algorithm.

Currently implemented tasks are:

- **Funnel board**:  
    ![Funnel board animation](img/fubo_demo.gif)  
    Task: Given first frame of the trajectory, predict platform where the ball will land.

- **Room runner**:  
    ![Room runner animation](img/rr_demo.gif)  
    Task: Given the first frame of the trajectory, predict color of the room in which the green dot will end up.


## Dependencies

- box2d==2.3.2
- cairocffi==0.8.0
- gizeh=0.1.10
- imageio==2.1.2
- moviepy==0.2.3.2
- numpy==1.14.0
- pillow==5.0.0
- tqdm==4.11.2
- tensorflow==1.5.0



## Generate datasets

Room runner:  
`python -m generate.generate_dataset --dataset rr --seed 1234 --n_trajectories 500 --output_dir /path/to/dataset/directory/`

Funnel board:  
`python -m generate.generate_dataset --dataset fubo --seed 1234 --n_trajectories 500 --output_dir /path/to/dataset/directory/`

Use `--n_processes N` to use N parallel workers (results in nondeterministic ordering of examples).
