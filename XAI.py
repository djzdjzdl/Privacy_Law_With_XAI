from matplotlib.pyplot import text
import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.ensemble
from sklearn.metrics import classification_report
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
import matplotlib as plot
import numpy as np

#Add AI if you want
class Do_AI:

    def __init__(self, x_train, y_train, x_test, y_test):     
        Do_AI.RFClassifier(x_train, y_train, x_test, y_test)

    def RFClassifier(x_train, y_train, x_test, y_test):
        model = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        print(classification_report(y_test, pred))
        return model, pred

class Do_XAI:

    ti = TfidfVectorizer(stop_words='english', max_features=1000, lowercase=False)

    def __init__(self):
        x_train, y_train, x_test, y_test = Do_XAI.Get_Dataset('./Resume.csv')
        model, pred = Do_AI.RFClassifier(Do_XAI.ti.fit_transform(x_train), y_train, Do_XAI.ti.fit_transform(x_test), y_test)
        Do_XAI.Get_Lime(x_test, make_pipeline(Do_XAI.ti, model))

    def Get_Dataset(file_path):
        train_set, test_set = datasets.Fetch_Dataset(file_path, 1, 3, 0.2).Returner()
        return np.array(train_set).T[0], np.array(train_set).T[1], np.array(test_set).T[0], np.array(test_set).T[1]

    def Get_Lime(text_datasets, pipelines):
        Do_XAI.Do_Lime(LimeTextExplainer(class_names=[0, 1]), text_datasets, pipelines, 1)

    def Do_Lime(XAI, text_datasets, pipelines, idx):
        exp = XAI.explain_instance(text_datasets[idx], pipelines.predict_proba, num_features=20)
        exp.as_list()
        fig = exp.as_pyplot_figure()
        fig.savefig('./Analyze_{}.jpg'.format(idx))

if __name__ == "__main__":
    Do_XAI()