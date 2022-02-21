# Usages

The following scripts should be run under the `src` folder.

## Prepare utility data
We first need to build some helper data, like the vocab, graph, etc.
```shell
# sample the negative samples, and build the training pairs
python -m scripts.sample_train --fsize L

# build entity vocab and knowledge graph
python -m scripts.build_vocab_and_graph --fsize L

# build the dict of user historical clicked news
python -m scripts.build_hist_dict --fsize L --fmark train
python -m scripts.build_hist_dict --fsize L --fmark dev
python -m scripts.build_hist_dict --fsize L --fmark test

# build the dict of news features
python -m scripts.build_news_dict --fsize L
``` 


## Build dataset
```shell
# build training examples
python -m scripts.build_training_examples --fsize L

# build validate examples
python -m scripts.build_eval_examples --fsize L
``` 

Since the test set is very large, we split it to multiple parts.
```shell
# split the test behaviors data into seven parts
python -m scripts.split_test_data

# build examples based on each part
python -m scripts.build_test_examples --fsplit p0
python -m scripts.build_test_examples --fsplit p1
python -m scripts.build_test_examples --fsplit p2
python -m scripts.build_test_examples --fsplit p3
python -m scripts.build_test_examples --fsplit p4
python -m scripts.build_test_examples --fsplit p5
python -m scripts.build_test_examples --fsplit p6
```


## Training
We use 4 GPUs to train the model.
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python training.py training.gpus=4
```

At the end of each epoch, we will conduct the validation on the dev set.
We can select the best model to conduct the testing based on the validation performance.

## Testing
Use the trained model to predict the results for each test set part.
```shell
bash test.sh
```

Then, merge the results of all test parts.
```shell
cat ../data/L/result/p* > ../data/L/result/all_pred.result
```

Next, generate the submission file required by MIND contest platform.
```shell
python gen_submission.py

# Compress it
zip prediction.zip prediction.txt
```

Finally, the file `prediction.zip` can be submitted to the official website of the MIND contest to obtain the performance results of our model GREP. 