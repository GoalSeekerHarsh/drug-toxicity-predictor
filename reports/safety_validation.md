# Safety Validation Report

## Random Split
- Hazard precision: **1.0000**
- Hazard recall: **0.0920**
- Validated coverage rate: **0.9771**
- Out-of-envelope rate: **0.0229**

## Scaffold Split
- Hazard precision: **0.8478**
- Hazard recall: **0.1018**
- Validated coverage rate: **0.9614**
- Out-of-envelope rate: **0.0386**

## Safety Policy
- No hard `SAFE` verdict is allowed outside the validated envelope.
- Priority dictionary matches still bypass directly to `CRITICAL HAZARD`.

## Failure Examples
- False negative candidate: `C[C@@H]1CC[C@@]2(OC1)O[C@H]1[C@@H](O)[C@H]3[C@@H]4CC[C@H]5C[C@@H](O[C@@H]6O[C@H](CO)[C@H](O[C@@H]7O[C@H](CO)[C@@H](O)[C@H](O[C@@H]8OC[C@@H](O)[C@H](O)[C@H]8O)[C@H]7O[C@@H]7O[C@H](CO)[C@@H](O)[C@H](O[C@@H]8O[C@H](CO)[C@@H](O)[C@H](O)[C@H]8O)[C@H]7O)[C@H](O)[C@H]6O)[C@H](O)C[C@]5(C)[C@H]4CC[C@]3(C)[C@H]1[C@@H]2C` | prob=0.6525 | verdict=UNCERTAIN | in_envelope=False
- False negative candidate: `Oc1ccc(/N=N/c2ccccc2)cc1` | prob=0.6415 | verdict=UNCERTAIN | in_envelope=True
- False negative candidate: `Clc1ccc(Nc2nnc(Cc3ccncc3)c3ccccc23)cc1` | prob=0.6408 | verdict=UNCERTAIN | in_envelope=True
- False negative candidate: `S=C(Nc1ccccc1)Nc1ccccc1` | prob=0.6382 | verdict=UNCERTAIN | in_envelope=True
- False negative candidate: `C[C@]12CCC(=O)C=C1CC[C@@H]1[C@@H]2CC[C@]2(C)C(=O)CC[C@@H]12` | prob=0.6376 | verdict=UNCERTAIN | in_envelope=True
