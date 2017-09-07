## Result

FB15k:

|Model|MeanRank(Raw)|MeanRank(Filter)|Hit@10(Raw)|hit@10(Filter)|
|-----|:-----------:|:--------------:|:---------:|:------------:|
|TransE(n=50, rounds=1000)|204|69|43.78%|63.71%|
|TransE(n=50, rounds=1000, no-norm)|203|66|45.84%|66.03%|
|TransE(n=100, rounds=1000)|219|70|49.00%|74.72%|
|TransE(n=100, rounds=1000, no-norm)|216|67|50.83%|76.74%|
|TransD(n=50, rounds=1000, test\_E)|209|83|44.82%|64.12%|
|TransD(n=50, rounds=1000, test\_E, no-norm)|204|76|45.57%|65.09%|
|TransD(n=50, rounds=1000, test\_D)|214|77|44.10%|64.01%|
|TransD(n=50, rounds=1000, test\_D, no-norm)|210|71|45.98%|66.05%|
|TransD(n=100, rounds=1000, test\_E)|224|81|49.90%|76.05%|
|TransD(n=100, rounds=1000, test\_E, no-norm)|214|69|51.01%|76.96%|
|TransD(n=100, rounds=1000, test\_D)|231|82|49.10%|74.56%|
|TransD(n=100, rounds=1000, test\_D, no-norm)|224|74|50.84%|76.27%|

Result Analysis:

1. no-norm gets better results than norm by 1%~2% of hit@10.
2. TransD gets similar results to TransE, especially when we use test\_transE we even get better results than test\_transD which may prove the useless of TransD model.
