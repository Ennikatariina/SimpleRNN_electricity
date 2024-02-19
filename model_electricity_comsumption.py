import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.layers import Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error


def load_data (filename):
    data = pd.read_csv(filename, encoding='ISO-8859-1', delimiter=';', decimal=',')
    column_names = data.columns
    #print(column_names)
    drop_columns=['Status','Spot-keskihinta (snt/kWh, sis alv)', 'Kustannukset (yhteensä)','Perusmaksu (myynti)','Yksiaikainen kustannus (myynti)']
    for col in drop_columns:
        if col in column_names:
            data.drop(col, axis=1, inplace=True)
        else:
            print(f"Saraketta '{col}' ei löytynyt.")
    return data

#Time format (Tunti-column) is string. Let's changes the format ti datetime type.
def convert_time_column(dataframe, column_name):
    dataframe[column_name] = pd.to_datetime(dataframe[column_name])
    dataframe[column_name] = pd.to_numeric(dataframe[column_name], downcast='integer') // 10**9
    return dataframe



def init_model(input_shape,x_train, y_train):
      
    model = Sequential()
    
    model.add(SimpleRNN( input_shape=input_shape, units=4, activation='tanh'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    

    #Modify training parameters if needed
    model.fit(x_train, y_train, epochs=5, batch_size=1, verbose=2)
    return model
    


# Plot the result
def plot_result(trainY, testY, train_predict, test_predict):
    actual = np.append(trainY, testY)
    predictions = np.append(train_predict, test_predict)
    rows = len(actual)
    plt.figure(figsize=(15, 6), dpi=80)
    plt.plot(range(rows), actual)
    plt.plot(range(rows), predictions)
    plt.axvline(x=len(trainY), color='r')
    plt.legend(['Actual', 'Predictions'])
    plt.xlabel('Observation number after given time steps')
    plt.ylabel('Scale of electricity consumption')
    plt.title('The Red Line Separates The Training And Test Examples')

# Prepare the input X and target Y
def get_XY(dat, time_steps):
    Y_ind = np.arange(time_steps, len(dat), time_steps)
    Y = dat[Y_ind]
    print("tgttgttg",Y)
    rows_x = len(Y)
    X = dat[range(time_steps*rows_x)]
    print(X)
    X = np.reshape(X, (rows_x, time_steps, 2))    
    return X, Y

def print_error(trainY, testY, train_predict, test_predict):    
    # Error of predictions
    train_rmse = math.sqrt(mean_squared_error(trainY, train_predict))
    test_rmse = math.sqrt(mean_squared_error(testY, test_predict))
    # Print RMSE
    print('Train RMSE: %.3f RMSE' % (train_rmse))
    print('Test RMSE: %.3f RMSE' % (test_rmse)) 

if __name__ == "__main__":
    
    #Loading the data
    data = load_data('sahko.csv')
    
    #Let's changes the format to datetime type. 
    convert_time_column(data, 'Tunti')
    print(data.shape)
    
    #Nan values are removed
    data.dropna(inplace=True)
    print("nan ",data.shape)
    
    #Dubpicates are removed
    data.drop_duplicates(inplace=True)
    print("duplicates ",data.shape)
    print(data.head())
    print(data.columns)
    
    print(data)

   
    
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    print("dfgdfg",data)
 
    time_steps = 72
    
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps, :])
        y.append(data[i+time_steps, 1])
    X = np.array(X)
    y = np.array(y)
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    print("ytest",y_test.shape)
    model = init_model(input_shape=(time_steps,3),x_train=x_train, y_train=y_train)
    y_pred = model.predict(x_test)
    y_pred = y_pred.flatten()
    # make predictions
    train_predict = model.predict(x_train)
    
    print(y_pred)
    print(y_pred.shape)
   
    
    plot_result(y_train, y_test, train_predict, y_pred)
    print_error(y_train, y_test, train_predict, y_pred)
    
