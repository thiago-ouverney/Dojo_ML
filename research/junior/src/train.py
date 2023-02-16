import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer,make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn import linear_model
from sklearn.ensemble import VotingRegressor
import warnings 
#VARIABLES
warnings.filterwarnings("ignore")
seed = 42

#DATA EXTRACTION
df_raw = pd.read_csv('datasets/linear_model/fish.csv')


df_raw.columns = ['species', 'weight', 'vertical_length', 'diagonal_length', 'cross_length', 'height', 'width']
df_raw = df_raw.drop([142, 143, 144]).reset_index(drop=True)

# definição das variáveis dependentes e independentes do dataframe df_raw
raw_X_cru = df_raw.copy()
#raw_X = df_raw.drop(columns=['species', 'weight']).copy()
raw_y = df_raw['weight'].copy()

def get_log(x):
    return x

class DropInteligente(BaseEstimator, TransformerMixin):
    def __init__(self,colunas_drop:list,log:bool=False):
        super().__init__()
        self.colunas_drop =colunas_drop
        self.log = log
        pass
    def fit(self,x:pd.DataFrame, y:pd.Series=None):
        return self
    def transform(self,x:pd.DataFrame, y:pd.Series=None):
        df_raw = x.copy()
        raw_X = df_raw.drop(columns=self.colunas_drop).copy()
        if self.log:
            raw_X = get_log(raw_X)
        return raw_X



# criação de dados de treino e teste do dataframe df_raw
raw_X_train, raw_X_test, raw_y_train, raw_y_test = train_test_split(
    raw_X_cru,
    raw_y,
    test_size=0.3,
    random_state=42
)

#### MODELS

model= linear_model.Ridge()


#pipeline for columns transformations on categorical features
cat_preprocessing = make_pipeline( SimpleImputer(missing_values=np.nan, strategy='most_frequent'),
                                    OneHotEncoder(handle_unknown='ignore') 
                                    ) #Só vai fazer no test data o q já fez no train ou em inf no test)

num_preprocessing = make_pipeline( SimpleImputer(missing_values=np.nan, strategy='median'))

pipe_preprosseging = ColumnTransformer( [("numeric_transf", num_preprocessing, make_column_selector(dtype_exclude=object)),    # NOME-PROCESSSO  $$$$$ TRANFORMACAO A SER APLCIADA $$$$$ COLUNAS QUE VAO SOFRER A TRANF.
                                        ("categorical_transf", cat_preprocessing, make_column_selector(dtype_include=object))])

droper = DropInteligente(colunas_drop=['species', 'weight'])
#### PIPELINES
pipe = Pipeline(steps = [("Drop_Inteligente",droper),
                          ("Preprocessamento Padrão", pipe_preprosseging),
                          ("LinearModel", model )
                      ]
                      )


r1 = linear_model.Ridge(alpha=4, positive=True)
r2 = linear_model.Lasso(alpha=2, positive=True)
r3 = linear_model.LassoLars(alpha=0, positive=True)
model_voting = VotingRegressor([('Ridge',r1),
                                ('Lasso',r2),
                                ('LassoLars',r3)],
                                weights=[1,2,3]
                                )

pipeVoting = Pipeline(steps = [("Drop_Inteligente",droper),
                          ("Preprocessamento Padrão", pipe_preprosseging),
                          ("Voting", model_voting )
                      ]
                      )


parameters = {
    "Drop_Inteligente__colunas_drop": [['species', 'weight'],['weight']]
    ,"Voting__weights": [[0,0,1],[1,1,1],[1,1,0]],
    "Voting__Ridge__alhpa": []
    #"LinearModel__alpha": [1,2,3,4]
}


pipe_grid = GridSearchCV(pipeVoting,parameters,verbose=True,cv=5)
pipe_grid.fit(raw_X_cru,raw_y)

print(pipe_grid.score(raw_X_test,raw_y_test))
# print(pipe_grid.best_estimator_)
print(pipe_grid.best_params_)
print(pipe_grid.best_score_)


