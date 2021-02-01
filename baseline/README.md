MNLI Transferred Models
===========================================
We modify the code from [the text classification code from transformers examples](https://github.com/huggingface/transformers/tree/master/examples/text-classification).


### Train
The training commands are as follows: 

```python
# train MnliBert
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./roberta_bert_electra.py --model_name_or_path bert-base-cased --do_train --do_eval --max_seq_length 512 --per_device_train_batch_size 8 --learning_rate 2e-5 --num_train_epochs 15 --output_dir OUTPUT_DIR --train_file TRAIN_FILE --validation_file VALIDATION_FILE --cache_dir CACHE_DIR

# train MnliRoberta
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./roberta_bert_electra.py --model_name_or_path roberta-base --do_train --do_eval --max_seq_length 512 --per_device_train_batch_size 8 --learning_rate 2e-5 --num_train_epochs 15 --output_dir OUTPUT_DIR --train_file TRAIN_FILE --validation_file VALIDATION_FILE --cache_dir CACHE_DIR

# train MnliElectra
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./roberta_bert_electra.py --model_name_or_path google/electra-base-discriminator --do_train --do_eval --max_seq_length 512 --per_device_train_batch_size 8 --learning_rate 2e-5 --num_train_epochs 15 --output_dir OUTPUT_DIR --train_file TRAIN_FILE --validation_file VALIDATION_FILE --cache_dir CACHE_DIR
```
Rplace the OUTPUT_DIR, TRAIN_FILE, VALIDATION_FILE, CACHE_DIR with your own files and directories. OUTPUT_DIR is the directory to save trained models, CACHE_DIR is the directory to save auto downloaded transformers pretrained models.

Moreover, TRAIN_FILE, VALIDATION_FILE are files end with ".csv" and the seperator of each column needs to be \t. It should contain three columns with column names premise hypothesis and label (the first row of file needs to be premise\thypothesis\tlabel).
### Test
The testing commands are as follows:

```python
CUDA_VISIBLE_DEVICES=0 python ./roberta_bert_electra.py --model_name_or_path CHECKPOINT_DIR --do_eval --do_predict --max_seq_length 512 --output_dir OUTPUT_DIR --train_file TRAIN_FILE --validation_file VALIDATION_FILE --cache_dir CACHE_DIR
```
Rplace the OUTPUT_DIR, TRAIN_FILE, VALIDATION_FILE, CACHE_DIR with your own files and directories. OUTPUT_DIR here is the directory to save evaluation output.

