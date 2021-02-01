# AdvFact
## Diagnostic evaluation set
* Base evaluation datasets
    - DocAsClaim: Document sentence as claim.
    - RefAsClaim: Reference summary sentence as claim.
    - FaccTe: Human annotated evaluation set from [Evaluating the Factual Consistency of Abstractive Text Summarization](https://www.aclweb.org/anthology/2020.emnlp-main.750.pdf)
    - QagsC: Human annotated evaluation set from [Asking and Answering Questions to Evaluate the Factual Consistency of Summaries](https://www.aclweb.org/anthology/2020.acl-main.450.pdf)
    - RankTe: Human annotated evaluation set from [Ranking Generated Summaries by Correctness: An Interesting but Challenging Application for Natural Language Inference](https://www.aclweb.org/anthology/P19-1213.pdf)
    - FaithFact: Human annotated evaluation set from [On Faithfulness and Factuality in Abstractive Summarization](https://www.aclweb.org/anthology/2020.acl-main.173.pdf)
* Adversarial transformation
    - Antonym Substitution
    - Numerical Editing
    - Entity Replacement
    - Syntactic Pruning
Every adversarial transformation can be performed on the six base evaluation datasets, thus results in 24 diagnostic evaluation set. All base evaluation datasets and diagnostic evaluation sets can be found [here]().
## Augmented training data
The augmented training data can be downloaded [here]().
## Factuality metrics
    - FactCC: the codes and original FactCC can be downloaded from [](https://github.com/salesforce/factCC). The four FactCCs trained with sub sampling and augmented data can be down loaded from [here]().
    - Dae: the codes and trained model can be downloaded from [](https://github.com/tagoyal/dae-factuality).
    - BertMnli, RobertaMnli, ElectraMnli codes are included in xxx and the trained models can be downloaded [here]().
    - Feqa: the codes and trained model can be downloaded from [](https://github.com/esdurmus/feqa).
## Error analysis samples
The samples that are misclassified by the FactCC can be downloaded [here]()