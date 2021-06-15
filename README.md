# Learning Set Functions that are Sparse in Non-Orthogonal Fourier Bases

## Abstract 

Many applications of machine learning on discrete domains, such as learning preference functions in recommender systems or auctions, can be reduced to estimating a set function that is sparse in the Fourier domain. In this work, we present a new family of algorithms for learning Fourier-sparse set functions. They require at most nk - k log(k) + k queries (set function evaluations), under mild conditions on the Fourier coefficients, where is the size of the ground set and the number of non-zero Fourier coefficients. In contrast to other work that focused on the orthogonal Walsh-Hadamard transform, our novel algorithms operate with recently introduced non-orthogonal Fourier transforms that offer different notions of Fourier-sparsity. These naturally arise when modeling, e.g., sets of items forming substitutes and complements. We demonstrate effectiveness on several real-world applications.

## Bibtex

```bibtex
@article{Wendler_Amrollahi_Seifert_Krause_Püschel_2021, 
title={Learning Set Functions that are Sparse in Non-Orthogonal {F}ourier Bases}, 
volume={35}, 
url={https://ojs.aaai.org/index.php/AAAI/article/view/17232}, 
number={12}, 
journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
author={Wendler, Chris and Amrollahi, Andisheh and Seifert, Bastian and Krause, Andreas and P{\"u}schel, Markus}, 
year={2021}, 
month={May}, 
pages={10283-10292}
}
```

# Implementation

We provide a sample implementation of our novel algorithms SSFT and SSFT+.

## Installation

Make sure you have at least Python 3.6. We ran everything on Python 3.8.

The auction simulation test suite requires pyjnius, which requires Cython.
```bash
pip install cython
```

Furthermore, download sats-v0.6.4.jar from https://github.com/spectrumauctions/sats/releases/tag/v0.6.4
and place it in ./exp/datasets/PySats/lib.

Now, you can install the remaining requirements.
```bash
pip install -r requirements.txt
```
If you run into trouble with pyjnius, please consult: https://pyjnius.readthedocs.io/en/stable/installation.html

### Note

*The python wrapper of the spectrum auction test suite (located in exp/datasets/PySats) is part of the SATS-project [1] and currently still in development. It was provided to us by the authors of [1].*


[1] http://spectrumauctions.org/

## Experiments

We use sacred to run our experiments. The -F flag specifies a target directory for logs and results (see metrics.json and other additional files in target_dir/run_id/). We store the learnt Fourier coefficients and Fourier support together with some statistics in the "result" field in run.json. In addition, intermediate results (relative errors, mean absolute errors, number of non-zero Fourier coefficients, number of queries, etc.) are logged into metrics.json. For the sensor placement experiment we additionally create a CSV and a PDF file containing the quality of the sensor placements obtained by greedily maximizing the true information gain function vs its Fourier-sparse approximation. 

### Sensor Placement

To run the sensor placement experiments execute:

```bash
python -m exp.run_sensorplacement with model.SSFT dataset.BERKELEY -F target_dir 
```

To get the results for the other datasets exchange dataset.BERKELEY with dataset.RAIN or dataset.CALIFORNIA respectively.


### Preference Elicitation in Combinatorial Auctions


To run the preference elicitation example run:

```bash

python -m exp.run_elicitation with model.SSFTPlus dataset.MRVM repetitions=1 -F target_dir

```

It is possible to run a smaller auction (with 18 goods) by exchanging dataset.MRVM with dataset.GSVM.
