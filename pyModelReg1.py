import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import save_model
from matplotlib import pyplot as plt
import seaborn as sns
import csv as csv
import cv2
import os
import sys

# The following lines adjust the granularity of reporting. 
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

TOTAL_WIDTH = 288.00;
TOTAL_HEIGHT = 190.00;
TOTAL_PIXELS = TOTAL_WIDTH * TOTAL_HEIGHT;
TOTAL_COLORS = 16777216.00;
loaded_model =  False;


def load_data_norm():
 print ('loading and normalizing data')

 train_df = pd.read_csv('C:/Users/jesus/source/repos/pyModelReg1/training.csv');
 test_df = pd.read_csv('C:/Users/jesus/source/repos/pyModelReg1/test.csv');

 train_df = train_df.astype(float)
 test_df =  test_df.astype(float);

 print (train_df['Center'][0])
 print (test_df['Center'][0])

 train_df['Center'] = train_df['Center'] / TOTAL_PIXELS;
 test_df['Center'] = test_df['Center'] / TOTAL_PIXELS;

 print (train_df['Center'][0])
 print (test_df['Center'][0])

 originalPoint = train_df['Center'][0] * TOTAL_PIXELS;
 print (originalPoint)

 originalPoint = test_df['Center'][0] * TOTAL_PIXELS;
 print (originalPoint)

 for i in range(0,342):
    colName = 'Tile'+str(i);
    train_df[colName] = train_df[colName] / TOTAL_COLORS;
    test_df[colName] = test_df[colName] / TOTAL_COLORS;

 print (train_df['Tile0'][0])
 print (test_df['Tile0'][0]) 

 originalTile = train_df['Tile0'][0] * TOTAL_COLORS;
 print (originalTile);

 originalTile = test_df['Tile0'][0] * TOTAL_COLORS;
 print (originalTile); 

 train_df = train_df.reindex(np.random.permutation(train_df.index)); # shuffle the examples

 originalTile = train_df['Tile6'][344] * TOTAL_COLORS;
 print (originalTile);

 originalTile = test_df['Tile0'][0] * TOTAL_COLORS;
 print (originalTile);

 originalTile = train_df['Center'][344] * TOTAL_COLORS;
 print (originalTile);

 print ('data norm finished');

 return train_df, test_df;

def get_feature_layer():

 feature_columns = []

 for i in range(0,342):
        feature_columns.append(tf.feature_column.numeric_column("Tile"+str(i),shape=(1,),default_value=None,dtype=tf.dtypes.float64,normalizer_fn=None));
        
 my_feature_layer = tf.keras.layers.DenseFeatures(feature_columns);
 return my_feature_layer;

def plot_the_loss_curve(epochs, mse):
  """Plot a curve of loss vs. epoch."""

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Mean Squared Error")

  plt.plot(epochs, mse, label="Loss")
  plt.legend()
  plt.ylim([mse.min()*0.95, mse.max() * 1.03])
  plt.show()  

print("Defined the plot_the_loss_curve function.")

def create_model2(my_learning_rate, my_feature_layer):
  """Create and compile a simple linear regression model."""
  # Most simple tf.keras models are sequential.
  model = tf.keras.models.Sequential()

  # Add the layer containing the feature columns to the model.
  model.add(my_feature_layer)

  # Describe the topography of the model by calling the tf.keras.layers.Dense
  # method once for each layer. We've specified the following arguments:
  #   * units specifies the number of nodes in this layer.
  #   * activation specifies the activation function (Rectified Linear Unit).
  #   * name is just a string that can be useful when debugging.

  # Define the first hidden layer with 20 nodes.   
  model.add(tf.keras.layers.Dense(units=24, 
                                  activation='relu',                                   
                                  name='Hidden1'))
  
  # Define the second hidden layer with 12 nodes. 
  model.add(tf.keras.layers.Dense(units=12, 
                                  activation='relu',                                   
                                  name='Hidden2'))

  
  # Define the output layer.
  model.add(tf.keras.layers.Dense(units=1,  
                                  name='Output'))                              
  
  model.compile(optimizer=tf.keras.optimizers.Adam(lr=my_learning_rate),
                loss="mean_squared_error",
                metrics=[tf.keras.metrics.MeanSquaredError()])

  return model

def train_model(model, dataset, epochs, batch_size, label_name):
  """Feed a dataset into the model in order to train it."""

  # Split the dataset into features and label.  
  features = {name:np.array(value) for name, value in dataset.items()}
  label = np.array(features.pop(label_name))  
  model.fit(x=features, y=label, batch_size=batch_size, epochs=epochs, shuffle=True)
  #history = model.fit(dataset, epochs=15)

  # Get details that will be useful for plotting the loss curve.
  #epochs = history.epoch
  #hist = pd.DataFrame(history.history)
  #rmse = hist["mean_squared_error"]

  return model;



def do_train_eval(learning_rate,epochs,batch_size,label_name,train_df,test_df,feature_layer):
    
# Establish the model's topography.
 my_model = create_model2(learning_rate, feature_layer)

# Train the model on the normalized training set.
 train_model(my_model, train_df, epochs, batch_size, label_name);
 #plot_the_loss_curve(epochs, mse);

 #test_features = {name:np.array(value) for name, value in test_df.items()}
 #test_label = np.array(test_features.pop(label_name)) # isolate the label
 #print("\n Evaluate the linear regression model against the test set:")
 #my_model.evaluate(x = test_features, y = test_label, batch_size=batch_size)

 return my_model;

def loadModel():
 learning_rate = 0.005
 epochs = 1
 batch_size = 100
 label_name = "Center"  

 print('Loading model');

 train_df,test_df = load_data_norm();
 feature_layer = get_feature_layer(); 
 model = do_train_eval(learning_rate,epochs,100,'Center',train_df,test_df,feature_layer);

 print(model.summary());

 print('Loading weights');
 model.load_weights('C:/Users/jesus/source/repos/pyModelReg1/ws.h5')

 return model;


def predict(model,dataset,batch_size): 
  # Split the dataset into features and label.  
  features = {name:np.array(value) for name, value in dataset.items()}  
  pred_y = model.predict(x=features, batch_size=batch_size, verbose=1)  

  return pred_y

def load_predict_norm(inputsFile):
  val_df = pd.read_csv(inputsFile);  
  print(val_df['Tile40'][0]);    
  val_df = val_df / TOTAL_COLORS;
  print(val_df['Tile40'][0]);    
  return val_df;


def do_predict(model,inputsFile):
    val_df = load_predict_norm(inputsFile)
    
    y_val = predict(model,val_df,1)
    print('y_val:' + str(y_val[0]));
    return y_val[0];


def predict_point(model,inputsFile):

    head, tail = os.path.split(inputsFile);   
    line = '';

    original_point_predict = do_predict(model,inputsFile);
    original_point_predict = original_point_predict * TOTAL_PIXELS;
    pointE1 = original_point_predict * 0.008146455278620124;
    point = original_point_predict + pointE1;
    py = point / TOTAL_WIDTH;
    px = point % TOTAL_WIDTH;    

    print('original point: ' + str(original_point_predict));
    print('error: ' + str(pointE1));
    print ('point: ' + str(point));
    print(str(px)+","+str(py));

    pfName = tail.split('.')[0] + '_p.csv';

    pointFile = open(head+'/'+pfName,'w');
    line = str(point)+','+str(px)+','+str(py);
    pointFile.write(line);
    pointFile.close();

    return point, px,py;
  
  
def main_entry():
    if (len(sys.argv)<1):
        print('set path');
        return

    dir_path = sys.argv[1];
    dir_path = dir_path.replace('\\','/')
    print('Predicting for:' + dir_path);
    processDir(dir_path);
    print('Predicting done for:' + dir_path);

def processDir(dir_path):

    inputFiles = [f for f in os.listdir(dir_path) if os.path.isfile(dir_path+'/'+f) and f.split('.')[1] == 'png'];
    model1 = loadModel();

    for inputFile in inputFiles:
        head, tail = os.path.split(inputFile);    
        fileName = tail.split('.')[0];
        dirForFile = dir_path+'/'+fileName;
        isdir = os.path.isdir(dirForFile)
        inputCsv = fileName + '.csv';
        inputPath= dirForFile+'/'+inputCsv
        if (isdir!=False):         
            print('Predicting points for: ' + inputFile)
            predict_point(model1,inputPath);
        else:
            print('path not found: ' + dirForFile)

main_entry();





