from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
import pandas as pd
import numpy as np


def tf_idf(
        # Основные параметры
        train,
        oos=None,
        oot=None,
        text_field=None,
        target_name=None,

        # Параметры TfidfVectorizer()
        params=None
):

    if oos is None:
        oos = pd.DataFrame(data=np.zeros(len(train.columns)).reshape(1, len(train.columns)), columns=train.columns)

    if oot is None:
        oot = pd.DataFrame(data=np.zeros(len(train.columns)).reshape(1, len(train.columns)), columns=train.columns)

    tf_idf_vectorizer = TfidfVectorizer(**params)

    train_tf_idf_vector = tf_idf_vectorizer.fit_transform(train[text_field].values.astype('str'))
    oos_tf_idf_vector = tf_idf_vectorizer.transform(oos[text_field].values.astype('str'))
    oot_tf_idf_vector = tf_idf_vectorizer.transform(oot[text_field].values.astype('str'))

    # преобразуем feature name в цифры для обучения, так как русские символы не воспринимаются
    number_feature_list = [i for i in range(len(tf_idf_vectorizer.get_feature_names()))]

    train_tf_idf = pd.DataFrame(data=train_tf_idf_vector.toarray(), columns=number_feature_list, index=train.index)
    oos_tf_idf = pd.DataFrame(data=oos_tf_idf_vector.toarray(), columns=number_feature_list, index=oos.index)
    oot_tf_idf = pd.DataFrame(data=oot_tf_idf_vector.toarray(), columns=number_feature_list, index=oot.index)

    if target_name:
        y_train = train[target_name]
        y_oos = oos[target_name]
        y_oot = oot[target_name]

        train_tf_idf = pd.concat([train_tf_idf, y_train], axis=1)
        oos_tf_idf = pd.concat([oos_tf_idf, y_oos])
        oot_tf_idf = pd.concat([oot_tf_idf, y_oot])

    return train_tf_idf, oos_tf_idf, oot_tf_idf
