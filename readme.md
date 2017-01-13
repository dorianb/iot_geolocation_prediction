
# Prédiction de la position d'objets connectés

Les objets connectés envoient à des intervalles de temps réguliers des messages qui tansitent par le réseau GSM. Des antennes appelées stations de base reçoivent ces messages et sont chargées de les transmettre via un réseau filaire. Notre objectif est d'implémenter un algorithme d'apprentissage automatique pour prédire la position d'un objet à partir des messages réceptionnés par les stations de base.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import math
from scipy.stats import gaussian_kde
from geopy.distance import vincenty
import time

from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline 
from sklearn.model_selection import cross_val_predict, train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor

%matplotlib inline
```

## Chargement des données


```python
train = pd.read_csv("mess_train_list.csv")
test = pd.read_csv("mess_test_list.csv")
train_label = pd.read_csv("pos_train_list.csv")
```


```python
print("Training set:", train.shape)
train[:5]
```

    Training set: (39250, 8)





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>objid</th>
      <th>bsid</th>
      <th>did</th>
      <th>nseq</th>
      <th>rssi</th>
      <th>time_ux</th>
      <th>bs_lat</th>
      <th>bs_lng</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>573bf1d9864fce1a9af8c5c9</td>
      <td>2841</td>
      <td>473335.0</td>
      <td>0.5</td>
      <td>-121.5</td>
      <td>1.463546e+12</td>
      <td>39.617794</td>
      <td>-104.954917</td>
    </tr>
    <tr>
      <th>1</th>
      <td>573bf1d9864fce1a9af8c5c9</td>
      <td>3526</td>
      <td>473335.0</td>
      <td>2.0</td>
      <td>-125.0</td>
      <td>1.463546e+12</td>
      <td>39.677251</td>
      <td>-104.952721</td>
    </tr>
    <tr>
      <th>2</th>
      <td>573bf3533e952e19126b256a</td>
      <td>2605</td>
      <td>473335.0</td>
      <td>1.0</td>
      <td>-134.0</td>
      <td>1.463547e+12</td>
      <td>39.612745</td>
      <td>-105.008827</td>
    </tr>
    <tr>
      <th>3</th>
      <td>573c0cd0f0fe6e735a699b93</td>
      <td>2610</td>
      <td>473953.0</td>
      <td>2.0</td>
      <td>-132.0</td>
      <td>1.463553e+12</td>
      <td>39.797969</td>
      <td>-105.073460</td>
    </tr>
    <tr>
      <th>4</th>
      <td>573c0cd0f0fe6e735a699b93</td>
      <td>3574</td>
      <td>473953.0</td>
      <td>1.0</td>
      <td>-120.0</td>
      <td>1.463553e+12</td>
      <td>39.723151</td>
      <td>-104.956216</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("Train set position:", train_label.shape)
train_label[:5]
```

    Train set position: (39250, 2)





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lat</th>
      <th>lng</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39.606690</td>
      <td>-104.958490</td>
    </tr>
    <tr>
      <th>1</th>
      <td>39.606690</td>
      <td>-104.958490</td>
    </tr>
    <tr>
      <th>2</th>
      <td>39.637741</td>
      <td>-104.958554</td>
    </tr>
    <tr>
      <th>3</th>
      <td>39.730417</td>
      <td>-104.968940</td>
    </tr>
    <tr>
      <th>4</th>
      <td>39.730417</td>
      <td>-104.968940</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("Test set:", test.shape)
test[:5]
```

    Test set: (29286, 8)





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>objid</th>
      <th>bsid</th>
      <th>did</th>
      <th>nseq</th>
      <th>rssi</th>
      <th>time_ux</th>
      <th>bs_lat</th>
      <th>bs_lng</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>573be2503e952e191262c351</td>
      <td>3578</td>
      <td>116539.0</td>
      <td>2.0</td>
      <td>-111.0</td>
      <td>1.463542e+12</td>
      <td>39.728651</td>
      <td>-105.163032</td>
    </tr>
    <tr>
      <th>1</th>
      <td>573c05f83e952e1912758013</td>
      <td>2617</td>
      <td>472504.0</td>
      <td>0.0</td>
      <td>-136.0</td>
      <td>1.463551e+12</td>
      <td>39.779908</td>
      <td>-105.062479</td>
    </tr>
    <tr>
      <th>2</th>
      <td>573c05f83e952e1912758013</td>
      <td>3556</td>
      <td>472504.0</td>
      <td>0.0</td>
      <td>-127.0</td>
      <td>1.463551e+12</td>
      <td>39.780658</td>
      <td>-105.053676</td>
    </tr>
    <tr>
      <th>3</th>
      <td>573c05f83e952e1912758013</td>
      <td>3578</td>
      <td>472504.0</td>
      <td>0.0</td>
      <td>-129.0</td>
      <td>1.463551e+12</td>
      <td>39.728651</td>
      <td>-105.163032</td>
    </tr>
    <tr>
      <th>4</th>
      <td>573c05f83e952e1912758013</td>
      <td>4058</td>
      <td>472504.0</td>
      <td>0.0</td>
      <td>-105.0</td>
      <td>1.463551e+12</td>
      <td>39.783211</td>
      <td>-105.088747</td>
    </tr>
  </tbody>
</table>
</div>



## Exploration des données

### Nombre de stations recevant un même message


```python
data = train.groupby("objid").count()["bsid"]
data_with_3_pos = data[data>=3].count()
data_with_3_pos_less = data[data<3].count()
data_with_1_pos = data[data==1].count()

print("Messages reçus par 3 stations de base ou plus: %2.2f" %(data_with_3_pos / data.count() * 100), "%")
print("Messages reçus par moins de 3 stations: %2.2f" %(data_with_3_pos_less / data.count() * 100), "%")
print("Messages reçus par une station: %2.2f" %(data_with_1_pos / data.count() * 100), "%")

plt.figure(figsize=(12,9))
plt.hist(data, bins= 100)
plt.xlabel("Nombre de stations ayant reçues le même message")
plt.ylabel("Nombre de messages")
plt.xlim([1, 30])
plt.xticks(np.arange(1, 30+1, 1.0))
plt.show()
```

    Messages reçus par 3 stations de base ou plus: 62.39 %
    Messages reçus par moins de 3 stations: 37.61 %
    Messages reçus par une station: 21.90 %



![png](output_10_1.png)



```python
listOfBs = np.union1d(np.unique(train['bsid']), np.unique(test['bsid']))
print("Nombre de stations ayant reçues au moins un message %d" %(len(listOfBs)))
```

    Nombre de stations ayant reçues au moins un message 259



```python
plt.figure(figsize=(10, 5))
kernel = gaussian_kde(train["rssi"])
xx = np.linspace(-150, -80, 1000)
plt.plot(xx, kernel(xx))
plt.xlabel("RSSI")
plt.ylabel("Density")
_ = plt.title("Density of RSSI")
```


![png](output_12_0.png)



```python
plt.figure(figsize=(15, 9))
mp = Basemap(width=4000000, height=6000000, projection='lcc',
            resolution='i', lat_0=train["bs_lat"].mean(), lon_0=train["bs_lng"].mean())
mp.bluemarble()
mp.scatter(train["bs_lng"].values, train["bs_lat"].values,
       latlon=True,  marker='.',color='m')

_ = mp.drawmapscale(train["bs_lng"].min(), train["bs_lat"].min() - 10,
                 2 * train["bs_lng"].max() - train["bs_lng"].min(), train["bs_lat"].min() - 10,
                 500, fontsize = 14)
```


![png](output_13_0.png)


## Régression

Quelles caractéristiques chosir pour notre modèle ?

### Feature Matrix construction


#### Message received or not


```python
def feat_mat_const(train, listOfBs):
    
    messages = train["objid"].unique()
    df_feat = pd.DataFrame(index= messages, columns=listOfBs).fillna(0)
    
    # Pour chaque station recevant le message
    # Affecter la valeur du signal correspondant
    for message in messages:
        bsids = train[train["objid"] == message]["bsid"]
        df_feat.loc[message, bsids] = 1
    
    return df_feat
```

#### Message received with rssi or not


```python
def feat_mat_const(train, listOfBs):
    
    # Pour chaque station ne recevant pas de message
    # Affecter la plus petite valeur existante auquelle on retranchera 10
    min_rssi = train["rssi"].min() - 10000
    messages = train["objid"].unique()
    df_feat = pd.DataFrame(index= messages, columns=listOfBs).fillna(min_rssi)
    
    # Pour chaque station recevant le message
    # Affecter la valeur du signal correspondant
    for message in messages:
        bsids = train[train["objid"] == message]["bsid"]
        for bsid in bsids:
            df_feat.loc[message, bsid] = train[(train["objid"] == message) & (train["bsid"] == bsid)]["rssi"].iloc[0]
    
    return df_feat
```

#### Message received with location


```python
def feat_mat_const(train, listOfBs):
    
    min_lat = train["bs_lat"].mean()
    min_lng = train["bs_lng"].mean()
    messages = train["objid"].unique()
    columns = []
    list(map(lambda bs: columns.append("lat_" + str(bs)), listOfBs))
    list(map(lambda bs: columns.append("lng_" + str(bs)), listOfBs))
    
    df_feat = pd.DataFrame(index= messages, columns=columns)
    df_feat[df_feat.columns[:len(listOfBs)]] = df_feat[df_feat.columns[:len(listOfBs)]].fillna(min_lat)
    df_feat[df_feat.columns[len(listOfBs):]] = df_feat[df_feat.columns[len(listOfBs):]].fillna(min_lng)
    
    # Pour chaque station recevant le message
    for message in messages:
        bsids = train[train["objid"] == message]["bsid"]
        for bsid in bsids:
            lat = train[(train["objid"] == message) & (train["bsid"] == bsid)]["bs_lat"].iloc[0]
            lng = train[(train["objid"] == message) & (train["bsid"] == bsid)]["bs_lng"].iloc[0]
            df_feat.loc[message, ("lat_" + str(bsid))] = lat
            df_feat.loc[message, ("lng_" + str(bsid))] = lng
    
    return df_feat
```

#### Message received with location and rssi


```python
def feat_mat_const(train, listOfBs):
    
    min_rssi = train["rssi"].min() - 10000
    min_rssi_lat = train["bs_lat"].mean() * min_rssi
    min_rssi_lng = train["bs_lng"].mean() * min_rssi
    messages = train["objid"].unique()
    columns = []
    list(map(lambda bs: columns.append("lat_" + str(bs)), listOfBs))
    list(map(lambda bs: columns.append("lng_" + str(bs)), listOfBs))
    
    df_feat = pd.DataFrame(index= messages, columns= columns)
    df_feat[df_feat.columns[:len(listOfBs)]] = df_feat[df_feat.columns[:len(listOfBs)]].fillna(min_rssi_lat)
    df_feat[df_feat.columns[len(listOfBs):]] = df_feat[df_feat.columns[len(listOfBs):]].fillna(min_rssi_lng)
    
    # Pour chaque station recevant le message
    for message in messages:
        bsids = train[train["objid"] == message]["bsid"]
        for bsid in bsids:
            lat = train[(train["objid"] == message) & (train["bsid"] == bsid)]["bs_lat"].iloc[0]
            lng = train[(train["objid"] == message) & (train["bsid"] == bsid)]["bs_lng"].iloc[0]
            rssi = train[(train["objid"] == message) & (train["bsid"] == bsid)]["rssi"].iloc[0]
            df_feat.loc[message, ("lat_" + str(bsid))] = lat * rssi
            df_feat.loc[message, ("lng_" + str(bsid))] = lng * rssi
    
    return df_feat
```

La dernière proposition de construction de matrice d'observation est celle retenue par la suite.

### Ground truth construction


```python
def ground_truth_const(train, train_label):
    train_set = train.copy()
    train_set[["lat", "lng"]] = train_label
    ground_truth_lat = train_set.groupby("objid").mean()["lat"]
    ground_truth_lng = train_set.groupby("objid").mean()["lng"]
    return ground_truth_lat, ground_truth_lng
```


```python
t0 = time.time()
X_train = feat_mat_const(train, listOfBs)
X_test = feat_mat_const(test, listOfBs)

print("Built features matrix in %s seconds" %(time.time() - t0))

y_lat, y_lng = ground_truth_const(train, train_label)
```

    Built features matrix in 698.4292361736298 seconds



```python
print(X_train.shape)
print(X_test.shape)
print(y_lat.shape)
print(y_lng.shape)
X_train.head()
```

    (6068, 518)
    (5294, 518)
    (6068,)
    (6068,)





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lat_879</th>
      <th>lat_911</th>
      <th>lat_921</th>
      <th>lat_944</th>
      <th>lat_980</th>
      <th>lat_1012</th>
      <th>lat_1086</th>
      <th>lat_1092</th>
      <th>lat_1120</th>
      <th>lat_1131</th>
      <th>...</th>
      <th>lng_9936</th>
      <th>lng_9941</th>
      <th>lng_9949</th>
      <th>lng_10134</th>
      <th>lng_10148</th>
      <th>lng_10151</th>
      <th>lng_10162</th>
      <th>lng_10999</th>
      <th>lng_11007</th>
      <th>lng_11951</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>573bf1d9864fce1a9af8c5c9</th>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>...</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
    </tr>
    <tr>
      <th>573bf3533e952e19126b256a</th>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>...</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
    </tr>
    <tr>
      <th>573c0cd0f0fe6e735a699b93</th>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>...</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
    </tr>
    <tr>
      <th>573c1272f0fe6e735a6cb8bd</th>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>...</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
    </tr>
    <tr>
      <th>573c8ea8864fce1a9a5fbf7a</th>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>-5636.333280</td>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>-432146.003811</td>
      <td>...</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
      <td>1.025347e+06</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 518 columns</p>
</div>



### Scaling


```python
X_scaler = StandardScaler()
X_train = pd.DataFrame(X_scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(X_scaler.transform(X_test), columns=X_test.columns)
X_train.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lat_879</th>
      <th>lat_911</th>
      <th>lat_921</th>
      <th>lat_944</th>
      <th>lat_980</th>
      <th>lat_1012</th>
      <th>lat_1086</th>
      <th>lat_1092</th>
      <th>lat_1120</th>
      <th>lat_1131</th>
      <th>...</th>
      <th>lng_9936</th>
      <th>lng_9941</th>
      <th>lng_9949</th>
      <th>lng_10134</th>
      <th>lng_10148</th>
      <th>lng_10151</th>
      <th>lng_10162</th>
      <th>lng_10999</th>
      <th>lng_11007</th>
      <th>lng_11951</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.012838</td>
      <td>-0.012838</td>
      <td>-0.012838</td>
      <td>-0.022241</td>
      <td>-0.036334</td>
      <td>-0.119193</td>
      <td>-0.105664</td>
      <td>-0.072812</td>
      <td>-0.012838</td>
      <td>-1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.036334</td>
      <td>0.018158</td>
      <td>0.240966</td>
      <td>0.330334</td>
      <td>0.012838</td>
      <td>0.217879</td>
      <td>0.012838</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.012838</td>
      <td>-0.012838</td>
      <td>-0.012838</td>
      <td>-0.022241</td>
      <td>-0.036334</td>
      <td>-0.119193</td>
      <td>-0.105664</td>
      <td>-0.072812</td>
      <td>-0.012838</td>
      <td>-1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.036334</td>
      <td>0.018158</td>
      <td>0.240966</td>
      <td>0.330334</td>
      <td>0.012838</td>
      <td>0.217879</td>
      <td>0.012838</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.012838</td>
      <td>-0.012838</td>
      <td>-0.012838</td>
      <td>-0.022241</td>
      <td>-0.036334</td>
      <td>-0.119193</td>
      <td>-0.105664</td>
      <td>-0.072812</td>
      <td>-0.012838</td>
      <td>-1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.036334</td>
      <td>0.018158</td>
      <td>0.240966</td>
      <td>0.330334</td>
      <td>0.012838</td>
      <td>0.217879</td>
      <td>0.012838</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.012838</td>
      <td>-0.012838</td>
      <td>-0.012838</td>
      <td>-0.022241</td>
      <td>-0.036334</td>
      <td>-0.119193</td>
      <td>-0.105664</td>
      <td>-0.072812</td>
      <td>-0.012838</td>
      <td>-1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.036334</td>
      <td>0.018158</td>
      <td>0.240966</td>
      <td>0.330334</td>
      <td>0.012838</td>
      <td>0.217879</td>
      <td>0.012838</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.012838</td>
      <td>-0.012838</td>
      <td>-0.012838</td>
      <td>-0.022241</td>
      <td>-0.036334</td>
      <td>-0.119193</td>
      <td>9.460168</td>
      <td>-0.072812</td>
      <td>-0.012838</td>
      <td>-1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.036334</td>
      <td>0.018158</td>
      <td>0.240966</td>
      <td>0.330334</td>
      <td>0.012838</td>
      <td>0.217879</td>
      <td>0.012838</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 518 columns</p>
</div>



### Measure error


```python
def vincenty_vec(vec_coord):
    vin_vec_dist = np.zeros(vec_coord.shape[0])
    if vec_coord.shape[1] !=  4:
        print('ERROR: Bad number of columns (shall be = 4)')
    else:
        vin_vec_dist = [vincenty(vec_coord[m,0:2],vec_coord[m,2:]).meters for m in range(vec_coord.shape[0])]
    return vin_vec_dist
```


```python
# evaluate distance error for each predicted point
def Eval_geoloc(y_train_lat , y_train_lng, y_pred_lat, y_pred_lng):
    vec_coord = np.array([y_train_lat, y_train_lng, y_pred_lat, y_pred_lng])
    err_vec = vincenty_vec(np.transpose(vec_coord))
    
    return err_vec
```

### Select hyper parameters


```python
def regressor_and_predict_CV(reg, parameters, df_feat, ground_truth_lat, ground_truth_lng, df_test):
    
    # train regressor and make prediction in the train set
    # Input: df_feat, ground_truth_lat, ground_truth_lng, df_test
    # Output: y_pred_lat, y_pred_lng

    X_train = np.array(df_feat)
    reg = GridSearchCV(reg, parameters, cv=10)
    
    reg.fit(X_train, ground_truth_lat)
    y_pred_lat = reg.predict(df_test)
    print("Best latitude estimator:", reg.best_estimator_)

    reg.fit(X_train, ground_truth_lng)
    y_pred_lng = reg.predict(df_test)
    print("Best longitude estimator:", reg.best_estimator_)
    
    return y_pred_lat, y_pred_lng
```

#### Régression linéaire basique


```python
def regressor_and_predict(df_feat, ground_truth_lat, ground_truth_lng, df_test):
    parameters = {
        'fit_intercept': [True, False]
    }
    """poly = PolynomialFeatures(2)
    print(df_feat.shape)
    df_feat = poly.fit_transform(df_feat)
    df_test = poly.fit_transform(df_test)
    print(df_feat.shape)"""
    reg = linear_model.LinearRegression(n_jobs=-1)
    return regressor_and_predict_CV(reg, parameters, df_feat, ground_truth_lat, ground_truth_lng, df_test)
```

#### Forêt aléatoire


```python
def regressor_and_predict(df_feat, ground_truth_lat, ground_truth_lng, df_test):
    parameters = {
        'criterion': ["mse"]
    }
    reg = RandomForestRegressor(n_estimators=1000, max_depth=10, n_jobs=-1)
    
    return regressor_and_predict_CV(reg, parameters, df_feat, ground_truth_lat, ground_truth_lng, df_test)
```

#### Extra Forêt


```python
def regressor_and_predict(df_feat, ground_truth_lat, ground_truth_lng, df_test):
    parameters = {
        'max_depth': [1, 5, 10, 20]
    }
    reg = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
    return regressor_and_predict_CV(reg, parameters, df_feat, ground_truth_lat, ground_truth_lng, df_test)
```


```python
random_seed = 13
X_train_b, X_val, y_train, y_val = train_test_split(X_train, np.transpose([y_lat, y_lng]),
                                                    train_size=0.9, random_state=random_seed)

# Learning on a part of the training set
# Predicting on another small part of the training set
y_pred_lat, y_pred_lng = regressor_and_predict(X_train_b, y_train[:, 0], y_train[:, 1], X_val)

# Measuring error on the small part of the training set not used for learning
err_vec = Eval_geoloc(y_val[:, 0] , y_val[:, 1], y_pred_lat, y_pred_lng)
```

    Best latitude estimator: RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=10,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_split=1e-07, min_samples_leaf=1,
               min_samples_split=2, min_weight_fraction_leaf=0.0,
               n_estimators=1000, n_jobs=-1, oob_score=False,
               random_state=None, verbose=0, warm_start=False)
    Best longitude estimator: RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=10,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_split=1e-07, min_samples_leaf=1,
               min_samples_split=2, min_weight_fraction_leaf=0.0,
               n_estimators=1000, n_jobs=-1, oob_score=False,
               random_state=None, verbose=0, warm_start=False)


### Model validation


```python
def regressor_and_predict(df_feat, ground_truth_lat, ground_truth_lng):
    
    X_train = np.array(df_feat)
    
    # Multiple output regressor
    reg = MultiOutputRegressor(RandomForestRegressor(n_estimators=1000, n_jobs=-1))
    y = pd.concat([ground_truth_lat, ground_truth_lng], axis=1)
    y_pred = cross_val_predict(reg, df_feat, y, cv=10)
    y_pred_lat = y_pred[:, 0]
    y_pred_lng = y_pred[:, 1]
    
    # Multiple regressors
    """reg = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
    y_pred_lat = cross_val_predict(reg, df_feat, ground_truth_lat, cv=10)
    y_pred_lng = cross_val_predict(reg, df_feat, ground_truth_lng, cv=10)"""
    
    return y_pred_lat, y_pred_lng

# Learning and predicting on full training set
y_pred_lat, y_pred_lng = regressor_and_predict(X_train, y_lat, y_lng)
# Measuring error on the full training set
err_vec = Eval_geoloc(y_lat, y_lng, y_pred_lat, y_pred_lng)
```

### Distribution of errors


```python
values, base = np.histogram(err_vec, bins=50000)
cumulative = np.cumsum(values)
plt.figure()
plt.plot(base[:-1]/1000, cumulative / np.float(np.sum(values))  * 100.0, c='blue')
plt.grid()
plt.xlabel('Distance Error (km)')
plt.ylabel('Cum proba (%)')
plt.axis([0, 30, 0, 100])
_ = plt.title('Error Cumulative Probability')
```


![png](output_47_0.png)



```python
print("Critère de prédiction: %2.2f km" %(np.percentile(err_vec, 80) / 1000))
```

    Critère de prédiction: 3.54 km



```python
# Données de tests et prédictions
plt.figure(figsize=(15, 9))
mp = Basemap(width=4000000, height=6000000, projection='lcc',
            resolution='i', lat_0=y_lat.mean(), lon_0=y_lng.mean())
mp.bluemarble()
mp.scatter(y_lng.values, y_lat.values, latlon=True, marker='.', color='y', label="Mesure")
mp.scatter(y_pred_lng, y_pred_lat, latlon=True, marker='.', color='m', label="Prédiction")

plt.legend(loc="best")
_ = mp.drawmapscale(y_lng.min(), y_lat.min() - 10, 2 * y_lng.max() - y_lng.min(),
                    y_lat.min() - 10, 500, fontsize = 14)
```


![png](output_49_0.png)


## Résultat


```python
def regressor_and_predict(df_feat, ground_truth_lat, ground_truth_lng, df_test):
    
    X_train = np.array(df_feat)
    reg = RandomForestRegressor(n_estimators=1000, n_jobs=-1)    
    
    reg.fit(X_train, ground_truth_lat)
    y_pred_lat = reg.predict(df_test)

    reg.fit(X_train, ground_truth_lng)
    y_pred_lng = reg.predict(df_test)
    
    return y_pred_lat, y_pred_lng

y_pred_lat, y_pred_lng = regressor_and_predict(X_train, y_lat, y_lng, X_test)
```


```python
# Données de tests et prédictions
plt.figure(figsize=(15, 9))
mp = Basemap(width=4000000, height=6000000, projection='lcc',
            resolution='i', lat_0=y_lat.mean(), lon_0=y_lng.mean())
mp.bluemarble()
mp.scatter(y_lng.values, y_lat.values, latlon=True, marker='.', color='y', label="Mesure")
mp.scatter(y_pred_lng, y_pred_lat, latlon=True, marker='.', color='m', label="Prédiction")

plt.legend(loc="best")
_ = mp.drawmapscale(y_lng.min(), y_lat.min() - 10, 2 * y_lng.max() - y_lng.min(),
                    y_lat.min() - 10, 500, fontsize = 14)
```


![png](output_52_0.png)



```python
test_res = pd.DataFrame(np.array([y_pred_lat, y_pred_lng]).T, columns = ['lat', 'lng'])
test_res.to_csv('pred_pos_test_list.csv', index=False)

test_res.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lat</th>
      <th>lng</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39.717294</td>
      <td>-105.088569</td>
    </tr>
    <tr>
      <th>1</th>
      <td>39.783026</td>
      <td>-105.071437</td>
    </tr>
    <tr>
      <th>2</th>
      <td>39.689990</td>
      <td>-105.005084</td>
    </tr>
    <tr>
      <th>3</th>
      <td>39.795613</td>
      <td>-105.074901</td>
    </tr>
    <tr>
      <th>4</th>
      <td>39.687741</td>
      <td>-105.001905</td>
    </tr>
  </tbody>
</table>
</div>


