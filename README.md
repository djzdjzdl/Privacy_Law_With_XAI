# Privacy_Law_With_XAI

## OverView

- Analyze Curriculum Vitae using XAI
- Using Kaggle Dataset [https://www.kaggle.com/gauravduttakiit/resume-dataset]
- This Code was created to provide the basis for AI to judge Resume.



### Directory Setting

- pdf [Extracted Dataset File]
- word [Extracted Dataset File]
- main.py [run XAI & Data Converter]
- XAI.py [XAI model implementation]
- datasets.py [Make Train_set, Test_set using csv file]



### How to Run

```
python main.py
```



## Function Explaination

### main.py [Run AI Model + XAI model]

- Just import XAI.py, then run Do_XAI function



### XAI.py [Train AI, Analyze with XAI]

- Do_AI [Train AI model with datasets]
  - RFClassifier [Implementation Random Forest Model]



- Do_XAI [Get text Analyze with LIME]
  - Get_Dataset [Get dataset using file path]
  - Get_Lime [Setting XAI Model with LIME]
  - Do_Lime [Get Analyzed data using XAI model]



### datasets.py [Setting train, testset using csv file]

- Fetch_Dataset(file_path, number_of_string_column, number_of_label_column, split_proportion) [Get train, testset]
  - Make_Train_Set(dataset, split_proportion) [Set Train, Testset with split proportion]
  - Dataset_Maker_With_Each_Label(csv_reader, number_of_string_column, number_of_label_column) [Making all_in_one dataset with each label values]
  - Dataset_Maker(csv_reader, number_of_string_column) [Making all_in_one dataset with random label values 0, 1]

