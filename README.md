#Transformer Based Multimodal Speech Emotion Recognition with Improved Neural Networks.

https://ieeexplore.ieee.org/document/9520692 
Link to the published article.


##Basic Idea:

With the procession of technology, the human-machine interaction research field is in growing need of robust automatic emotion recognition systems. Building machines that interact with humans by comprehending emotions paves the way for developing systems equipped with human-like intelligence. Previous architecture in this field often considers RNN models. However, these models are unable to learn in-depth contextual features intuitively. This paper proposes a transformer-based model that utilizes speech data instituted by previous works, alongside text and mocap data, to optimize our emotional recognition system’s performance. Our experimental result shows that the proposed model outperforms the previous state-of-the-art. The IEMOCAP dataset supported the entire experiment.

-----

##Preparation
###Dataset
You may execute our code by requesting the dataset from (https://sail.usc.edu/iemocap/) or by assessing our processed data from (https://drive.google.com/file/d/19GcLs3k-xB1R0y1JfX14Z46uPmNKJaLX/view?usp=share link). If you choose the processed data, unzip it and store it in the data folder upon downloading. However, if you obtain the full data from (https://sail.usc.edu/iemocap/), unzip it and analyze it using the file (scripts/data collected.ipynb).

The dataset contains roughly ten emotions. Following previous research, our research focused just on four emotions (anger, neutral, excitement, and sadness), therefore the processed data file only comprises these four emotions. You must process the data yourself if you want to increase the number of emotions detected in your project.

###How to run file:
After Data is processed (NEXT):

1. Run any of these files (Audio_speech, TextBERT, MotionCap)
	* Audio_speech: for training on speech only 
	* TextBERT: for training on text only
	* MotionCap: for training on motion capture(Hand_MotionCap, Head_MotionCap, FacialRotation_MotionCap)
	* Concatenation: for training all three modalities (speech + text + motion cap)

2. Extra files can also be executed for comparison purposes (Pairwise_Concatenation & Text_GLOVE_840B)
	* Pairwise_Concatenation: combines two modalities instead of 3.
	* Text_GLOVE: uses GLOVE distributed word representation model instead of BERT.

###Where to run the code: 
The code can be executed on Google Colab, Kaggle, or create your own environment on your machine. 

##Reference

Please refer to our article for more details.

@INPROCEEDINGS{9520692,  author={Patamia, Rutherford Agbeshi and Jin, Wu and Acheampong, Kingsley Nketia and Sarpong, Kwabena and Tenagyei, Edwin Kwadwo},  booktitle={2021 IEEE 2nd International Conference on Pattern Recognition and Machine Learning (PRML)},   title={Transformer Based Multimodal Speech Emotion Recognition with Improved Neural Networks},   year={2021},  volume={},  number={},  pages={195-203},  doi={10.1109/PRML52754.2021.9520692}}
