# KnowGQA
Question Answering model with SQuAD2.0. (Stanfard University CS224N default project)
The starter code is forked from https://github.com/minggg/squad

## 1.Introduction
In this work, we proposed KnowGQA to integrate extra knowledge graph information into the question answering system using GCN and knowledge attention. We retrieve the knowledge graph data for each word entity and its related edges from ConceptNet[1] and build knowledge sub-graph for each context and generate knowledge representation using GCN. Such representation is then be integrated into question answering system using knowledge attention. The knowledge attention mechanism can be used in any question answering system. Here, we combine knowledge attention with QANet train the combined model in SQuAD 2.0 dataset. Results show that the extra knowledge extensively improve the performance of QANet in SQuAD 2.0, which prove the ability of our framework.

## 2. Run the code
First, **cd** to repository file and type ` conda env create -f environment.yml ` to create conda environment.</br>
Run `source activate squad `.</br>
Run `python setup.py`.</br>
We integrate four model in the this project: BiDAF_nochar, BiDAF[2], QANet[3] and KnowGQA. The BiDAF_nochar is the default BiDAF without char embedding. To test each model, type following:</br>
BiDAF_nochar: `python trian.py -n=BiDAF_onchar --model_name=BiDAF_nochar --hidden_size=100`</br>
BiDAF: `python trian.py -n=BiDAF --model_name=BiDAF --hidden_size=100`</br>
QANet: `python trian.py -n=QANet --model_name=QANet --hidden_size=128 --h=8 --batch_size=32`</br>
KnowGQA:`python trian.py -n=KnowGQA --model_name=KnowGQA --hidden_size=96 --h=1 --batch_size=32`</br>

## 3. Model Structure

## 4. Results


## Acknowledgment 
Thank for Standard University and teacher group of CS224N for providing such wonderful course and online-free material. It really helps me learn both fundamental and state-of-art techniques in NLP field, which also inspired me the interest to explore more in this field. Meanwhile, thanks for the starter code in this default project.

## Reference
[1]R. Speer, J. Chin, and C. Havasi, ConceptNet 5.5: An Open Multilingual Graph of General Knowledge. 2019.</br>
[2]M. Seo, A. Kembhavi, A. Farhadi, and H. Hajishirzi, “Bidirectional Attention Flow for Machine Comprehension,” Nov. 2016, Accessed: Aug. 22, 2020. [Online]. Available: https://arxiv.org/abs/1611.01603.
[3]A. W. Yu et al., “QaNet: Combining local convolution with global self-attention for reading comprehension,” 2018.
