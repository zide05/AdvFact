# AdvFact
The directory contains trained models, diagnostic test sets and augmented training data for paper **Factuality Checker is not Faithful: Adversarial Meta-evaluation of Factuality in Summarization**

## Factuality metrics
Six representative factuality checkers included in the paper are as follows: 
* **FactCC:** the codes and original FactCC can be downloaded from [https://github.com/salesforce/factCC](https://github.com/salesforce/factCC). The four FactCCs trained with sub sampling and augmented data can be downloaded from [here](https://drive.google.com/drive/u/1/folders/1wg9jHrO90_t85ymRFBi7l6o4U7_fij_s).
* **Dae:** the codes and trained model can be downloaded from [https://github.com/tagoyal/dae-factuality](https://github.com/tagoyal/dae-factuality).
* **BertMnli, RobertaMnli, ElectraMnli:** the codes are included in [baseline](./baseline) and the trained models can be downloaded [here](https://drive.google.com/drive/u/1/folders/1wg9jHrO90_t85ymRFBi7l6o4U7_fij_s).
* **Feqa:** the codes and trained model can be downloaded from [https://github.com/esdurmus/feqa](https://github.com/esdurmus/feqa).

The table below represents the 6 factuality metrics and their model types as well as training datas.
<table class="tg">
<thead>
  <tr>
    <th class="tg-za14">Models</th>
    <th class="tg-za14">Type</th>
    <th class="tg-za14">Train data</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-za14">MnliBert</td>
    <td class="tg-za14">NLI-S</td>
    <td class="tg-za14">MNLI</td>
  </tr>
  <tr>
    <td class="tg-za14">MnliRoberta</td>
    <td class="tg-za14">NLI-S</td>
    <td class="tg-za14">MNLI</td>
  </tr>
  <tr>
    <td class="tg-za14">MnliElectra</td>
    <td class="tg-za14">NLI-S</td>
    <td class="tg-za14">MNLI</td>
  </tr>
  <tr>
    <td class="tg-za14">Dae</td>
    <td class="tg-za14">NLI-A</td>
    <td class="tg-za14">PARANMT-G</td>
  </tr>
  <tr>
    <td class="tg-za14">FactCC</td>
    <td class="tg-za14">NLI-S</td>
    <td class="tg-za14">CNNDM-G</td>
  </tr>
  <tr>
    <td class="tg-za14">Feqa</td>
    <td class="tg-za14">QA</td>
    <td class="tg-za14">QA2D,SQuAD</td>
  </tr>
</tbody>
</table>

**The model type and training data of factuality metrics.** NLI-A and NLI-S represent the model belongs to NLI-based metrics while defining facts as dependency arcs and span respectively. PARANMT-G  and CNNDM-G mean the automatically generated training data from PARANMT and CNN/DailyMail.


## Adversarial transformation codes
The codes of adversarial transformations are in the directory of [adversarial transformation](./adversarial transformation). To make adversarial transformation, please run the following commands:
```python
CUDA_VISIBLE_DEVICES=0 python main.py -path DATA_PATH -save_dir SAVE_DIR -trans_type all
```
Change the DATA_PATH and SAVE_DIR to your own data path and save directory.

## Diagnostic evaluation set
Six base evaluation datasets and four adversarial transformations are included in the paper.
* **Base evaluation datasets**
    - **DocAsClaim:** Document sentence as claim.
    - **RefAsClaim:** Reference summary sentence as claim.
    - **FaccTe:** Human annotated evaluation set from [Evaluating the Factual Consistency of Abstractive Text Summarization](https://www.aclweb.org/anthology/2020.emnlp-main.750.pdf)
    - **QagsC:** Human annotated evaluation set from [Asking and Answering Questions to Evaluate the Factual Consistency of Summaries](https://www.aclweb.org/anthology/2020.acl-main.450.pdf)
    - **RankTe:** Human annotated evaluation set from [Ranking Generated Summaries by Correctness: An Interesting but Challenging Application for Natural Language Inference](https://www.aclweb.org/anthology/P19-1213.pdf)
    - **FaithFact:** Human annotated evaluation set from [On Faithfulness and Factuality in Abstractive Summarization](https://www.aclweb.org/anthology/2020.acl-main.173.pdf)
* **Adversarial transformation**
    - Antonym Substitution
    - Numerical Editing
    - Entity Replacement
    - Syntactic Pruning

Every adversarial transformation can be performed on the six base evaluation datasets, thus results in 24 diagnostic evaluation set. All base evaluation datasets and diagnostic evaluation sets can be found [here](https://drive.google.com/drive/u/1/folders/1inYZnSkxj1JfgHHpR2OjfNXpT-SFc24p). The detailed information for *6 baseline test sets and 24 diagnostic sets* is shown in the table below :
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow" rowspan="2">Base Test Sets</th>
    <th class="tg-c3ow" colspan="4">Origin</th>
    <th class="tg-c3ow" colspan="4">Adversarial   Transformation</th>
  </tr>
  <tr>
    <td class="tg-c3ow">Dataset type</td>
    <td class="tg-c3ow">Nov.</td>
    <td class="tg-c3ow">#Sys.</td>
    <td class="tg-c3ow">#Sam.</td>
    <td class="tg-c3ow">AntoSub</td>
    <td class="tg-c3ow">NumEdit</td>
    <td class="tg-c3ow">EntRep</td>
    <td class="tg-c3ow">SynPrun</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">DocAsClaim</td>
    <td class="tg-c3ow">CNNDM </td>
    <td class="tg-c3ow">0 .0 </td>
    <td class="tg-c3ow">0</td>
    <td class="tg-c3ow">11490</td>
    <td class="tg-c3ow">26487</td>
    <td class="tg-c3ow">25283</td>
    <td class="tg-c3ow">6816</td>
    <td class="tg-c3ow">9533</td>
  </tr>
  <tr>
    <td class="tg-0pky">RefAsClaim</td>
    <td class="tg-c3ow">CNNDM </td>
    <td class="tg-c3ow">77.7</td>
    <td class="tg-c3ow">0</td>
    <td class="tg-c3ow">10000</td>
    <td class="tg-c3ow">14131</td>
    <td class="tg-c3ow">11621</td>
    <td class="tg-c3ow">28758</td>
    <td class="tg-c3ow">4572</td>
  </tr>
  <tr>
    <td class="tg-0pky">FaccTe</td>
    <td class="tg-c3ow">CNNDM </td>
    <td class="tg-c3ow">54</td>
    <td class="tg-c3ow">10</td>
    <td class="tg-c3ow">503</td>
    <td class="tg-c3ow">670</td>
    <td class="tg-c3ow">515</td>
    <td class="tg-c3ow">440</td>
    <td class="tg-c3ow">245</td>
  </tr>
  <tr>
    <td class="tg-0pky">QagsC</td>
    <td class="tg-c3ow">CNNDM </td>
    <td class="tg-c3ow">28.6</td>
    <td class="tg-c3ow">1</td>
    <td class="tg-c3ow">504</td>
    <td class="tg-c3ow">711</td>
    <td class="tg-c3ow">615</td>
    <td class="tg-c3ow">539</td>
    <td class="tg-c3ow">351</td>
  </tr>
  <tr>
    <td class="tg-0pky">RankTe</td>
    <td class="tg-c3ow">CNNDM </td>
    <td class="tg-c3ow">52.5</td>
    <td class="tg-c3ow">3</td>
    <td class="tg-c3ow">1072</td>
    <td class="tg-c3ow">1646</td>
    <td class="tg-c3ow">1310</td>
    <td class="tg-c3ow">767</td>
    <td class="tg-c3ow">540</td>
  </tr>
  <tr>
    <td class="tg-0pky">FaithFact</td>
    <td class="tg-c3ow">XSum</td>
    <td class="tg-c3ow">99.2</td>
    <td class="tg-c3ow">5</td>
    <td class="tg-c3ow">2332</td>
    <td class="tg-c3ow">363</td>
    <td class="tg-c3ow">94</td>
    <td class="tg-c3ow">114</td>
    <td class="tg-c3ow">118</td>
  </tr>
</tbody>
</table>

**The detailed statistics of baseline (left) and diagnostic (right) test sets.** For baseline test sets in the left, dataset type means the dataset that source document and summary belong to. Here, CNNDM means CNN/DailyMail dataset. Nov.(%) means the proportion of trigrams in claims that don't exist in source documents. #Sys. and #Sam. represent the number of summarization systems that the output summaries come from and the test set size respectively. For diagnostic test sets on the right, all cells mean the sample size of the sets.

## Error analysis samples
The 140 samples that are misclassified by the FactCC are in the directory: [data](./data)

## Augmented training data
The augmented training data can be downloaded [here](https://drive.google.com/drive/u/1/folders/1lrqfrubEhRECjHM9SooeGABJ4-FW5bAR).


