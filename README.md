# bsp-ilqg

## Introduction
Belief Space Motion Planning Using iLQG. Built on top of iLQG Matlab implementation by Yuval Tassa and
the paper "Motion Planning under Uncertainty using Iterative Local Optimization in Belief Space", Van den berg et al., 
International Journal of Robotics Research, 2012

Modified for the thesis Decentralized Planning and Control in Belief Space for the Human-Machine Interaction in Multi-Robot Scenarios
Xin Ye, Dec., 2020

## How To
For single robot planning without variable measurement quality, run:
```bash
run_single_robot.m
```
For decentralized planning, run:
```bash
run_belief_admm.m 
```
For centralized planning, run:
```bash
run_centralized.m 
```
For decentralized control, run the following only after centralized planning is finished, when the variables of centralized planning are already stored in base-workspace:
```bash
run_sparse_feedback.m 
```
The costs of nominal trajectories of centralized and decentralized planning can be computed by running the following when the variables of both plannings are already stored in base-workspace:
```bash
test_cost.m
```

## License

Copyright (c) 2017, Saurav Agarwal
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the distribution
    * Neither the name of the Texas A&M University nor the names
      of its contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
