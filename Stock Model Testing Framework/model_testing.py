import time
import numpy as np
import pandas as pd
from pyts.decomposition import SingularSpectrumAnalysis

class model_testing():

    def __init__(self):

        try:
            from sklearn.preprocessing import MinMaxScaler
        except:
            print("Need to install SciKit Learn.\npip install sklearn")
            return
        try:
            import pandas as pd
        except:
            print("Need to install Pandas\npip install pandas")




        self.date_col = "date"

        self.model_params = {
            'num_layers': 256,
            'lb': 10,
            'epochs': 100,
            'activation': 'relu',
            'cols': [],
            'target': None,
            'patience': 3,
        }
        self.results = {
            "ensemble": {},
            "traditional": {}
        }

        self.actions = {
            "ensemble": {},
            "traditional": {}
        }

    def build_df(self,symbol,**kwargs):
        from pathlib import Path
        import os
        import pandas as pd

        action = 'daily'
        save_folder = 'stocks'
        reduce_volume = True
        months = -1
        return_df = False

        for key,value in kwargs.items():

            if key == 'save_folder':
                save_folder = value

            if key == 'reduce_volume':
                reduce_volume = value
            
            if key == 'months':
                months = value

            if key == 'return_df':
                return_df = value

        #//*** Get Working Directory
        current_dir = Path(os.getcwd()).absolute()

        save_dir = current_dir.joinpath(save_folder)

        input_filename = f"{symbol}_{action}.csv.zip"

        input_filepath = save_dir.joinpath(input_filename)
        
        if not Path.exists(input_filepath):
            print(f"{input_filepath}  does not exist. Quitting")
            return None

        #print(f"Reading dataframe from File: {input_filepath}")
        out_df = pd.read_csv(input_filepath)

        #//*** Sort by Date
        out_df = out_df.sort_values(by='date',ascending=True)
        
        #//*** Reset the index
        out_df = out_df.reset_index(drop=True)
        
        #//**** Reduce the size of the volume column, to make the number range more normal
        if reduce_volume:
            out_df['volume'] = out_df['volume'] / 10000 

        #//*** If number of months specified, keep that number of months.
        #//*** By default this process keeps the entire dataframe
        if months > 0:
            out_df['date'] = pd.to_datetime(out_df['date'])
            out_df = out_df[ out_df['date'] >= out_df['date'].max()-pd.DateOffset(months=months )]

        
        #//*** Convert the Date Column to datetime
        out_df['date'] = pd.to_datetime(out_df['date']).dt.date

        #//*** Return out_df or assigned out_df to self.df
        if return_df:
            return out_df
        else:
            self.df = out_df

    def build_model(self, **kwargs):

        # //*** Store Local Variables Here
        lv = {
            "model_type": "ensemble",
            "options": {},

        }

        # //*** Assign kwargs to local_kwargs
        for key, value in lv.items():
            # //*** key value in kwargs assign the kwargs value
            if key in kwargs.keys():
                lv[key] = kwargs[key]

        # //*** Parse options if it exists
        # //*** If key in options exists in self.model_params, update self.model_params with option value
        if "options" in kwargs.keys():
            print("Assigning:")
            for key, value in kwargs['options'].items():
                if key in self.model_params.keys():
                    print("\t", key, "=", value)
                    self.model_params[key] = value

        if lv['model_type'] == "ensemble":
            self.__build_ensemble_model(lv['options'])
            return

        if lv['model_type'] == "traditional":
            print("Building Test/Train split Model")
            self.__build_traditional_model(lv['options'])
            return

        print("Unknown Model Type:", lv['model_type'])
        print("Valid Model Types: ensemble, traditional")
        print("Quitting")
        return

    def __generate_model(self, num_features):

        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        tf.keras.backend.clear_session()
        model = keras.models.Sequential()
        model.add(layers.LSTM(
            self.model_params['num_layers'],
            activation=self.model_params["activation"],
            input_shape=(self.model_params['lb'], num_features)
        ))

        # //*** Hidden Layers Don't seem to be the answer
        # model.add(layers.LSTM(num_layers,activation=activation,return_sequences=True, input_shape=(lb,1)))
        # model.add(layers.LSTM(num_layers,activation=activation,return_sequences=True) )
        # model.add(layers.LSTM(num_layers,activation=activation) )

        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss='mse')

        return model

    # split a multivariate sequence into samples
    def __split_sequences(self, sequences, n_steps):
        import numpy as np

        X, y = list(), list()
        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the dataset
            if end_ix > len(sequences):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1, -1]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def __build_traditional_model(self, options):

        import numpy as np
        from sklearn.preprocessing import MinMaxScaler
        import tensorflow as tf

        # //******************************************
        # //******************************************
        # //*** Nested internal Private function.
        # //*** Why is this a good idea? Mostly for explicit scoping
        # //******************************************
        # //******************************************

        # Create a function to process the data into lb observations look back slices
        # and create the train test dataset (90-10)
        def build_LSTM_data(data, lb):
            X, Y = [], []
            for i in range(len(data) - lb):
                X.append(data[i:(i + lb), 0])
                Y.append(data[(i + lb), 0])
            return np.array(X), np.array(Y)

        # //******************************************
        # //******************************************

        print("Begin Build Traditional Model")

        # //*** Local values
        # //*** Keep local working values here. Keeping a dictionary makes it easy to expand the code later
        # //*** Values are collected together.
        class local_variable:
            def __init__(self):
                # self.lb = 10
                self.train_split = .9
                self.pa = 1,  # //*** Interval to predict ahead

                # //*** Initializing empty variables for good form's sake
                self.cols = None
                self.target = None
                self.raw_combined = []
                self.dataset = None
                self.scl = None
                self.x = None
                self.y = None
                self.x_train = None
                self.x_test = None

                # //**********************
                # //*** Process Test_Split
                # //**********************
                if "train_split" in options.keys():
                    self.train_split = options["train_split"]

                error_msg = "Fields - train_split must be a float > 0 &  <= 1"
                if isinstance(self.train_split, float) == False:
                    print(error_msg)
                elif self.train_split <= 0:
                    print(error_msg)
                elif self.train_split > 1:
                    print(error_msg)

        # //**********************
        # //*** END Local Variable
        # //**********************

        # //*** Hold local variables in a class for organization and readability(?)
        lv = local_variable()

        if 'pa' in options.keys():
            lv.pa = options['pa']

        if 'target' in options.keys():
            lv.target = options['target']
        else:
            print("Model Building Error: Need to define Target column to predict\nOptions = { target=column_name}")
            return

        # //*** by Default set actual_target to target
        lv.actual_target = lv.target

        if 'cols' in options.keys():
            lv.cols = options['cols']

        if 'train_split' in options.keys():
            lv.train_split = options['train_split']

        if 'days_to_model' in options.keys():
            lv.days_to_model = options['days_to_model']

        if 'actual_target' in options.keys():
            lv.actual_target = options['actual_target']

        # //*** Build y values All values minus the target value.
        # //*** Example if predict ahead = 1, length should be len(df) -1
        lv.y = self.df[lv.pa:][lv.target].values
        lv.y = lv.y.reshape((len(lv.y)), 1)
        # print(len(self.df),len(lv.y))
        # print(lv.y)

        # //*** Gather the raw values for the x columns
        for col in lv.cols:
            loop_val = self.df.iloc[:-lv.pa][col].values
            # print(len(lv.y),len(loop_val))
            loop_val = loop_val.reshape((len(loop_val)), 1)
            lv.raw_combined.append(loop_val)

        # //*** Add the last column which is the target column
        lv.raw_combined.append(lv.y)

        # for x in lv.raw_combined:
        #    print(x.shape)

        # //*** Combine the raw values into an LSTM compatible structure
        lv.raw_combined = np.hstack(lv.raw_combined)

        # print(lv.raw_combined)
        # print(lv.raw_combined.min(),lv.raw_combined.max())

        # //*** Initialize the Scalar
        lv.scl = MinMaxScaler()

        # /*** Fit the Scalar
        lv.scl.fit(lv.raw_combined)

        # //*** Transfor the dataset using the scalar
        lv.dataset = lv.scl.transform(lv.raw_combined)

        # //*** Split the Dataset into a traditional test/train split
        # lv.train_dataset, lv.test_dataset = lv.dataset[:int(lv.dataset.shape[0]*lv.train_split)],lv.dataset[int(lv.dataset.shape[0]*lv.train_split):]
        lv.train_dataset, lv.test_dataset = lv.dataset[:-(lv.days_to_model + self.model_params['lb'] - 1)], lv.dataset[
                                                                                                            -(
                                                                                                                        lv.days_to_model +
                                                                                                                        self.model_params[
                                                                                                                            'lb'] - 1):]

        # print(lv.dataset.shape,lv.train_dataset.shape,lv.test_dataset.shape)

        # convert into input/output
        lv.x_train, lv.y_train = self.__split_sequences(lv.train_dataset, self.model_params['lb'])
        lv.x_test, lv.y_test = self.__split_sequences(lv.test_dataset, self.model_params['lb'])
        # print(lv.x_test.shape, lv.y_test.shape)

        # print("Shapes:", lv.x_test.shape,lv.y_test.shape)

        # //*** Get the Model. Use a function to generate a clean model for each use
        model = self.__generate_model(len(lv.cols))

        model.summary()

        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=self.model_params['patience'], mode='auto')
        history = model.fit(
            lv.x_train,
            lv.y_train,
            epochs=self.model_params['epochs'],
            shuffle=False,
            callbacks=[callback]
        )

        print("Epochs Run:", len(history.history['loss']))
        # //*** Build a container to hold predictions with similar dimensions to minmaxscalar's dimensions
        lv.predictions_decoded = []
        lv.actuals = []
        # //*** Build a single array of predictions
        lv.p = model.predict(lv.x_test)
        # print("lv.p:",lv.p.shape)
        # print(lv.p)
        # print(lv.y_test.reshape(-1,1))
        # //*** Create multiple arrays of the predictions, up to the scalar number of features
        for x in range(lv.scl.n_features_in_):
            lv.predictions_decoded.append(lv.p)
            lv.actuals.append(lv.y_test.reshape(-1, 1))

        # //*** Restructure into an hstack, Like we did initially.
        lv.predictions_decoded = np.hstack(lv.predictions_decoded)
        lv.actuals = np.hstack(lv.actuals)

        # //*** Inverse transform to get the human readable predictions
        lv.predictions_decoded = lv.scl.inverse_transform(lv.predictions_decoded)
        lv.actuals = lv.scl.inverse_transform(lv.actuals)

        # //*** Hsplit, grab the last column and reshape to get a list
        lv.predictions_decoded = np.hsplit(lv.predictions_decoded, [1, len(lv.cols)])[-1].reshape(1, -1)[0]
        lv.actuals = np.hsplit(lv.actuals, [1, len(lv.cols)])[-1].reshape(1, -1)[0]

        # //*** Totally overthought the actuals. Why are we rerse transforming, when we can just get the original
        # //*** values from the df?
        # //*** Also, this update allows actuals to be a different column than the target column.
        # //*** This is helpful if a different indicator is used to predict another as is the case of SSA.
        lv.actuals = self.df[lv.actual_target].iloc[-len(lv.predictions_decoded):].values
        print("actuals: ", len(lv.actuals), len(lv.predictions_decoded))
        # print(lv.scl.inverse_transform(lv.y_test.reshape(-1,1)))
        self.results['traditional'] = {
            # //*** The Actual Values are the y_test values
            # //*** These are the last column of lv.test_decoded. There was much reshaping to get this
            # //*** The size of the hsplit is the size of lv.cols

            "predict_date": self.df['date'][len(lv.x_test) * -1:],
            "actual": lv.actuals,
            "predict": lv.predictions_decoded,

        }

        # try:
        #    playsound.playsound(sound_filename)
        # except:
        #    pass
        print("Build Traditional Done")
        return

        history = model.fit(lv.x_train, lv.y_train, epochs=self.model_params['epochs'], shuffle=False)
        return

    def __build_ensemble_model(self, options):
        import numpy as np
        from sklearn.preprocessing import MinMaxScaler
        import tensorflow as tf
        import time
        from IPython.display import clear_output

        # //******************************************
        # //******************************************
        # //*** Nested internal Private function.
        # //*** Why is this a good idea? Mostly for explicit scoping
        # //******************************************
        # //******************************************

        # Create a function to process the data into lb observations look back slices
        # and create the train test dataset (90-10)
        def build_LSTM_data(data, lb):
            X, Y = [], []
            for i in range(len(data) - lb):
                X.append(data[i:(i + lb), 0])
                Y.append(data[(i + lb), 0])
            return np.array(X), np.array(Y)

        # //******************************************
        # //******************************************

        print("Begin Build Ensemble Model")

        # //*** Local values
        # //*** Keep local working values here. Keeping a dictionary makes it easy to expand the code later
        # //*** Values are collected together.
        class local_variable:
            def __init__(self):
                self.lb = 10
                self.pa = 1
                self.days_to_model = 10
                self.disp = True

                # //*** Initializing empty variables for good form's sake
                self.target = None  # //*** Column to Predict, this becomes the Y values
                self.cols = None
                self.raw_combined = []
                self.x_test_raw_combined = []
                self.dataset = None
                self.x_test_dataset = None

                # self.close_1     = None
                # self.scl_close_1 = None
                self.scl = None
                self.x_train = None
                self.x_test = None

                self.xtest_predictions = None
                self.prediction = None

                # //*** Looping Time Variables
                self.start_time = None
                self.loop_time = None

                # //*** Looping variables
                self.max_loop = None
                self.max_index = None
                self.remain_loop = None
                self.cycle_time = None

        # //**********************
        # //*** END Local Variable
        # //**********************

        # //*** Hold local variables in a class for organization and readability(?)
        lv = local_variable()

        # //*** Parse Options into lv (Local Variables)
        # //*** Using If Statement for an easy code folding block
        if True:
            # //*** Update Days to Model, which may or may not be contained in key
            # //*** If it's a string, check if value = traditional
            # //*** Then check if there are traditional results
            # //*** If yes, then days to model to length of results['actual']
            # //*** Else use the defaults
            if 'days_to_model' in options.keys():

                if isinstance(options['days_to_model'], str):
                    if options['days_to_model'] == 'traditional':

                        if 'actual' in self.results['traditional'].keys():
                            lv.days_to_model = len(self.results['traditional']['actual'])

                # //*** If It's an integer use that value
                elif isinstance(options['days_to_model'], int):
                    lv.days_to_model = options['days_to_model']

            if 'pa' in options.keys():
                lv.pa = options['pa']

            if 'target' in options.keys():
                lv.target = options['target']
            else:
                print("Model Building Error: Need to define Target column to predict\nOptions = { target=column_name}")
                return

            if 'cols' in options.keys():
                lv.cols = options['cols']

            print("Days to Model:", lv.days_to_model)
            lv.max_loop = len(self.df) - lv.days_to_model
            print("Max Loop: ", lv.max_loop)
            lv.start_time = time.time()

            # //*** Reset results ensemble Object
            self.results['ensemble'] = {

                "actual": [],
                "predict": [],
                "predict_date": [],

            }

            if lv.target == None:
                print("Need to define the target field to predict. \noptions = {target=column_to_predict}")
                return
            if lv.cols == None:
                print(
                    "Need to define the column(s) to process for the multi-variate LSTSM\noptions = [cols=[col1,col2,col_n...]]")

        for loop_index in range(lv.days_to_model, 0, -1):
            print("=== REFERENCE ===")

            lv.raw_combined = []
            lv.x_test_raw_combined = []

            # //*** Get a slice of the whole data frame to match just this section to model
            loop_df = self.df.iloc[range(0, len(self.df) - loop_index)]

            print(loop_df)

            lv.loop_time = time.time()

            # //*** Generate the maximum index for this loop
            # lv.max_index = len(loop_df)-loop_index
            lv.max_index = len(loop_df)

            # //*** Build y values All values minus the target value.
            # //*** Example if predict ahead = 1, length should be len(df) -1
            lv.y_test = loop_df[lv.pa:][lv.target].values
            lv.y_test = lv.y_test.reshape((len(lv.y_test)), 1)
            # print("lv.y")
            # print(lv.y)

            # //*** Gather the raw values for the x columns
            for col in lv.cols:
                loop_val = loop_df.iloc[:-lv.pa][col].values
                print(len(lv.y_test), len(loop_val))
                loop_val = loop_val.reshape((len(loop_val)), 1)
                lv.raw_combined.append(loop_val)

            # //*** Add the last column which is the target column
            lv.raw_combined.append(lv.y_test)

            # lv.x_test_dataset = loop_df.iloc[-self.model_params['lb']:]

            for col in lv.cols:
                loop_val = loop_df.iloc[-lv.lb:][col].values
                loop_val = loop_val.reshape((len(loop_val)), 1)
                lv.x_test_raw_combined.append(loop_val)

            # //*** Add the Target Column
            lv.x_test_raw_combined.append(lv.y_test[-lv.lb:].reshape(lv.lb, 1))

            # //*** hstack x_test
            lv.x_test_raw_combined = np.hstack(lv.x_test_raw_combined)

            # //**** X_test

            print(lv.x_test_raw_combined)

            for x in lv.raw_combined:
                print(x.shape)

            # //*** Combine the raw values into an LSTM compatible structure
            lv.raw_combined = np.hstack(lv.raw_combined)
            # print("Raw Combined")

            # print(lv.raw_combined)
            #####print(lv.raw_combined.min(),lv.raw_combined.max())

            # //*** Initialize the Scalar
            lv.scl = MinMaxScaler()

            # /*** Fit the Scalar
            lv.scl.fit(lv.raw_combined)

            # //*** Transform the dataset using the scalar
            lv.dataset = lv.scl.transform(lv.raw_combined)
            lv.x_test_dataset = lv.scl.transform(lv.x_test_raw_combined)

            # print("DataSet")
            # print(lv.dataset)
            # Split Sequqnces for LSTM Processing
            lv.x_train, lv.y_train = self.__split_sequences(lv.dataset, self.model_params['lb'])

            print("Xtrain.shape: ", lv.x_train.shape)
            print(lv.x_test_dataset.shape)
            # //**** Remove Target column from the hstack, to run a proper prediction
            # //**** reshape to fit model shape.
            # lv.x_test = np.hsplit(lv.x_test_dataset,(lv.x_train.shape[2]-1,1))[0].reshape(1,lv.lb,lv.x_train.shape[2])
            # print("x_test_dataset")
            # print(lv.x_test_dataset)
            # print("hsplit")
            lv.x_test = np.hsplit(lv.x_test_dataset, (len(lv.cols), 1))[0].reshape((1, lv.lb, len(lv.cols)))

            print("x_test shape: ", lv.x_test.shape)

            # //*** Get the Model. Use a function to generate a clean model for each use
            model = self.__generate_model(len(lv.cols))

            model.summary()

            callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=self.model_params['patience'],
                                                        mode='auto')
            history = model.fit(
                lv.x_train,
                lv.y_train,
                epochs=self.model_params['epochs'],
                shuffle=False,
                callbacks=[callback]
            )

            # //*** Build a container to hold predictions with similar dimensions to minmaxscalar's dimensions
            lv.predictions_decoded = []
            lv.actuals = []

            # //*** Build a single array of predictions
            lv.p = model.predict(lv.x_test)

            # //*** Convert the prediction, to the same shape as the MinMaxScalar to get the float value

            # //*** Build an empty array using the shape of lv.dataset as a guide. This size will vary
            t = np.empty(lv.dataset.shape[-1])

            # //*** Fill the empty array with the prediction
            t.fill(lv.p.flatten()[0])

            # //*** Resave predictions in newly shaped array
            lv.p = t

            # //*** perform inverse transform, discard the top level array, get the last vale/column which will always
            # //*** Be the predictions.
            lv.p = round(lv.scl.inverse_transform(lv.p.reshape(1, -1))[0][-1], 2)

            # print("results:")
            # print(lv.p,self.df.iloc[lv.max_index][lv.target],self.df.iloc[lv.max_index]['date'])

            self.results['ensemble']['predict_date'].append(self.df.iloc[lv.max_index]['date'])
            self.results['ensemble']['predict'].append(lv.p)
            self.results['ensemble']['actual'].append(self.df.iloc[lv.max_index][lv.target])

            lv.remain_loop = loop_index
            lv.cycle_time = int(time.time() - lv.loop_time)

            clear_output(wait=True)
            print(lv.days_to_model - loop_index + 1, "/", lv.days_to_model, "[", lv.p, "] - ",
                  self.df.iloc[lv.max_index][lv.target], " ", lv.cycle_time, "s - remaining: ",
                  lv.cycle_time * lv.remain_loop, "s")
            if True:  # //*** Print as needed
                print("DF")
                print(self.df.iloc[loop_index:lv.max_index])

            continue

            lv.total_time = int(time.time() - lv.start_time)

        # try:
        #    playsound.playsound(sound_filename)
        # except:
        #    pass
        print("Build Ensemble Complete")

    def plot_predictions(self, **kwargs):
        from matplotlib import pyplot as plt

        extra_cols = []

        priority = []

        disp_params=True

        plot_type = 'all'

        for key, value in kwargs.items():
            if key == 'extra_cols':
                extra_cols = value

            if key == 'priority':
                priority = value

            if key == 'disp_params':
                disp_params = value

            if key == 'plot_type':
                plot_type = value

        #//*** Plot both types by default
        plotting = ['traditional','ensemble']

        #//*** Set Models to plat based on manual plot_type assignmnet
        if plot_type == "traditional" or plot_type == "t":
            result_type = 'traditional'
            plotting = ['traditional']


        if plot_type == "ensemble" or plot_type == "e":
            result_type = 'ensemble'
            plotting = ['ensemble']

        
        #//*** Plot each model type in result. If a model results don't exist, they will be skipped.
        for result_type in plotting:

            skip = True
            #//*** Verify Results exists to plot.
            #//*** If not... Skip
            for check in ["predict",'actual','predict_date']:

                #//*** If we find one field, we should have them all
                if check in self.results[result_type].keys():
                    skip = False
                    break

            #//*** Results not found. Skip plotting results
            if skip:
                continue


            mp = ""

            #for key, value in self.model_params.items():
            #    mp += f"{key}:{value} "

            mp += "Prediction Model Scores:\n"

            for key, value in self.score_models()[result_type].items():
                mp += f"{key}:{value} "
            mp += f"\nTarget: {self.model_params['target']}"

            lw = 1.5

            plot_actual = self.results[result_type]['actual']

            plot_predict = self.results[result_type]['predict']
            predict_date = self.results[result_type]['predict_date']

            plt.figure(figsize=(12, 8))
            plt.style.use('fivethirtyeight')

            lw = 1.5
            if 'actual' in priority:
                lw = 3


                    
            plt.plot(predict_date, plot_actual, label="Actual", linewidth=lw, color='steelblue')

            lw = 1.5
            if 'predict' in priority:
                lw = 3
            plt.plot(predict_date, plot_predict, label="Predicted", linewidth=lw, color='red')

            if plot_type == "traditional":
                i = self.results[plot_type]['predict_date'].index

                indexes = self.df.loc[i].index

            for col in extra_cols:
                lw = 1.5
                if col in priority:
                    lw = 3
                if plot_type == "traditional":

                    plt.plot(predict_date, self.df.loc[indexes][col], label=col, linewidth=lw)

                elif plot_type == "ensemble":

                    plt.plot(predict_date, self.df[col].iloc[-len(plot_predict):], label=col, linewidth=lw)

            # plt.xticks(ticks=range(len(plot_actual)),labels=predict_date, rotation = 30)
            plt.legend()

            if result_type == 'traditional':
                plt.title(f"{self.df['symbol'].unique()[0]}\nTraditional Test/Train Split\n{mp}")
            if result_type == 'ensemble':
                plt.title(f"{self.df['symbol'].unique()[0]}\nEnsemble Model\n{mp}")
            plt.show()

    def plot_cols(self, cols=None, **kwargs):
        from matplotlib import pyplot as plt

        # //*** If no Columns defined default to all columns
        if cols == None:
            cols = self.df.columns
        custom_exclude = []

        quick = True
        
        start_index = 0
        end_index = len(self.df)

        for key, value in kwargs.items():
            if key == 'start_index':
                start_index = value
            if key == 'end_index':
                end_index = value
            if key == 'quick':
                quick = value

            if key == 'exclude':
                custom_exclude = value

        # //*** Quick Plotting plots all columns except those listed in exclude
        # //*** The name says is all
        exclude = ['date', 'open', 'high', 'low', 'volume', 'symbol']

        plt.figure(figsize=(12, 8))
        plt.style.use('fivethirtyeight')

        first = True
        for col in self.df.columns:
            lw = 1.5
            if col in exclude:
                continue
            if col in custom_exclude:
                continue
            if first:
                lw = 4
                first = False
            plt.plot(self.df.iloc[start_index:end_index]['date'], self.df.iloc[start_index:end_index][col],
                     label=col, linewidth=lw)
        plt.title(f"{self.df['symbol'].unique()[0]}: Quick Plot")
        plt.legend()
        plt.show()

        
        """
            plt.figure(figsize=(12, 8))
            for col in cols:
                lw = 1.5
                if col == cols[0]:
                    lw = 3
                try:
                    plt.plot(self.df.iloc[start_index:end_index]['date'], self.df.iloc[start_index:end_index][col],
                             label=col, linewidth=lw)
                except:
                    pass
            plt.legend()
            plt.show()
        """


    def score_models(self):

        from sklearn.metrics import r2_score
        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import mean_absolute_error
        import math

        out = {}

        for model_type in self.results.keys():

            if 'actual' not in self.results[model_type].keys():
                continue

            actual = self.results[model_type]['actual']
            predict = self.results[model_type]['predict']

            out[model_type] = {
                "r2": round(r2_score(actual, predict), 2),
                "mae": round(float(mean_absolute_error(actual, predict)), 2),
                "mse": round(float(mean_squared_error(actual, predict)), 2),
                "rmse": round(math.sqrt(mean_squared_error(actual, predict)), 2)
            }
        return out

    # //*********************************************************************************************************************
    # //*** All Purpose function for generating actions based on predicted data.
    # //*** Action: action - Generates Buy/Hold/Sell signals (1/0/-1). Takes Actual Data, Prediction, Trade Threshold.
    # //******************   Uses iterrows() to explicitly calculate each action individually. Very deliberate choice
    # //******************   to avoid data leakage. Trade Threshold expressed as a percentage, is a minimum threshold
    # //******************   to execute a trade. Example: 5% Threshold a predicted price must change by 5% of the
    # //******************   current price to execute. If the current price is $1, a trade will only execute if the
    # //******************   predicted value is less than .95 or greater than 1.05. Higher thresholds indicate
    # //******************   more conservative and less volatile strategies. One threshold limits are met, there
    # //******************   is an explict decision tree of actions to take.
    # //*********************************************************************************************************************
    # //*** Action: accuracy - Generates a simple accuracy percentage from actual and predicted values. Compares actual to
    # //******************   predicted. Counts rows where the values are identical.
    # //*********************************************************************************************************************
    # //*** Action: pl -     Generates predicted Profit/Loss based on modeled trading strategy. Walks through each buy/hold/sell
    # //******************   action and simulates the profit/loss associated with the action. Each buy/sell/hold action is
    # //******************   explicitly handled. The resulting decision is based on the previousl buy/sell/hold action.
    # //******************   Example: A stock can't be sold if it hasn't been bought. This leads to 9 possible outcomes
    # //******************   for each action based on the previous buy/sell/hold action. The variable position holds the
    # //******************   current buy/sell/hold action as a trinary value 1/-1/0. Current_price is used to keep track
    # //******************   of whether a stock is currently held, and at what price.
    # //*********************************************************************************************************************
    def calc_results(self, action, **kwargs):

        threshold = 0
        actual_col = 'actual'
        predict_col = 'predict'
        action_col = 'actions'
        model_col = None
        get_list = False
        verbose = False
        model_type = None
        target = None
        action_type = None

        for key, value in kwargs.items():

            if key == 'threshold':
                threshold = value

            if key == 'actual':
                actual_col = value

            if key == 'predict':
                predict_col = value

            if key == 'model':
                model_col = value

            if key == 'get_list':
                get_list = value

            if key == 'verbose':
                verbose = value

            if key == 'model':
                model_type = value

            if key == 'target':
                target = value

            if key == 'action_type':
                action_type = value

            if key == 'verbose':
                verbose = value

        if verbose:
            print("BEGIN Calc Results")
        # print(self.df.iloc[-20:][ ['diff','ssa']  ])
        # print(self.results[model_type])

        if action == 'build_actions':

            # print("action type",action_type)
            tdf = pd.DataFrame()

            for col, value in self.results[model_type].items():
                tdf[col] = value

            current_price = tdf[actual_col].iloc[0]

            # input_df['diff'] = input_df[actual_col] - input_df[predict_col]

            out = []
            # //*** Loop through each element
            for row in tdf.iterrows():

                # //*** Calculate the Row Difference as an absolute value
                row_diff = abs(current_price - row[1][predict_col])

                row_thresh = current_price * threshold

                # //*** If the row_diff is less than the row_thresh. We are not predicting a huge price move
                # //*** Set position to 0, curent_price remains the same.
                if row_diff < row_thresh:
                    # //*** Set position to hold
                    out.append(0)
                    continue
                else:
                    # //*** Difference exceeds threshold Commit to buy sell action

                    if action_type == 'diff':
                        # //*** Base Actions on a relative difference measure.
                        # //*** If Value is positive = Buy
                        # //*** Negative Means Sell

                        if row[1][predict_col] > 0:

                            # //*** Set Position to buy
                            out.append(1)

                            # //*** Set the Current Price for Thresholding
                            current_price = row[1][actual_col]
                            continue

                        else:
                            # //*** Set Position to sell
                            out.append(-1)

                            # //*** Set the Current Price for Thresholding
                            current_price = row[1][actual_col]


                    else:
                        # //*** Calculate actions based on actual vs predicted values
                        # //*** Actual less than Predicted. Perform Buy
                        if row[1][actual_col] < row[1][predict_col]:

                            # //*** Set Position to buy
                            out.append(1)

                            # //*** Set the Current Price for Thresholding
                            current_price = row[1][actual_col]
                            continue

                        # //** Actual is greater than Predicted. Perform Sell
                        else:
                            # //*** Set Position to Sell
                            out.append(-1)

                            # //*** Set the Current Price for Thresholding
                            current_price = row[1][actual_col]
                            continue
            tdf['actions'] = out

            self.actions[model_type] = tdf
            # print("action table")
            # print(self.actions[model_type])
            return
            # //*** Actions at Threshold Built Return it
            # return (out)

        # //**************************************
        # //**************************************
        # //**************************************
        # //**** END action=='action'
        # //**************************************
        # //**************************************
        # //**************************************

        if action == 'pl':

            # self.actions[target] = self.df[target]
            if verbose:
                print("Begin pl")

            tdf = self.actions[model_type]
            # tdf[target] = self.df[target].iloc[-len(tdf):]

            indexes = self.df.iloc[-len(tdf):].index
            tdf.set_index(indexes, drop=True, inplace=True)
            tdf[target] = self.df[target]

            current_price = tdf.iloc[0][target]
            pl = 0
            if verbose:
                print("current_price", current_price)

            # pl -= current_price

            initial_price = current_price

            position = 1

            pl_list = []

            if action_col == None:
                print("action Column (-1,0,1) must be defined")
                print("model={column_name}")
                return

            if target == None:
                print("Target PriceColumn must be defined")
                print("actual={column_name}")
                return

            if verbose:
                print(f"Action Col: {action_col}")
                print(f"Target Col: {target}")

            for row in tdf.iterrows():

                if verbose:
                    print(row[1][action_col], row[1][target], current_price, pl, position)

                # //*** Sell Action
                if row[1][action_col] == -1:
                    if verbose:
                        print("---Sell: ", position, current_price, pl)

                    # //*** Perform action based on Position

                    # //*** We have already sold
                    # //*** Do Nothing and maintain Sell
                    if position == -1:
                        if verbose:
                            print("------Already Sold")
                        pl_list.append(pl)
                        continue

                    # //*** We Are holding From a Previous Position
                    # //*** Check Current Price, If we have a current Price, then sell
                    elif position == 0:
                        if verbose:
                            print("------Actualy Sell")

                        position = -1

                        # //*** We have Stock, Sell it
                        if current_price > 0:
                            pl += (row[1][target] - initial_price)

                            current_price = 0


                    # //*** We Previously Bought
                    # //*** Change position and Sell
                    elif position == 1:

                        position = -1

                        # //*** Double Check current_Price to verify we can sell
                        if current_price > 0:
                            if verbose:
                                print("------Actualy Sell")
                            pl += (row[1][target] - initial_price)

                            current_price = 0

                # //*** Hold Action
                # Basically Maintain the current Position
                elif row[1][action_col] == 0:
                    if verbose:
                        print("---Hold")
                    if position == -1:
                        position = 0

                    elif position == 0:
                        position = 0

                    elif position == 1:
                        position = 0


                # //*** Buy Action
                elif row[1][action_col] == 1:
                    if verbose:
                        print("---Buy")

                    # //*** Previously Sold
                    # //*** Let's Buy
                    if position == -1:

                        position = 1

                        # //*** Double Check current_price
                        if current_price == 0:

                            if verbose:
                                print("------Actually Buy -1")

                            current_price = row[1][target]

                            pl -= (row[1][target] - initial_price)

                    # //*** Currently Holding. Let's Buy if we havent already
                    elif position == 0:
                        position = 1

                        # //*** Double Check current_price
                        if current_price == 0:

                            if verbose:
                                print("------Actually Buy 0")

                            current_price = row[1][target]

                            pl -= (row[1][target] - initial_price)

                    # //*** We have already bought
                    elif position == 1:
                        position = 1

                        if verbose:
                            print("------Already Bought")

                        pl_list.append(pl)

                        continue
                pl_list.append(pl)

            if get_list == True:
                return pl_list

            return round(pl, 4)

        # //*** Generate the simple accuracy of the Actual vs predicted values
        if action == 'accuracy':
            if verbose:
                print(f"Actual: {actual_col} Predict: {predict_col}")
            return (input_df[actual_col] == input_df[predict_col]).astype(int)

        if action == 'r2':
            if verbose:
                print(f"r2 Actual: {actual_col} Predict: {predict_col}")

            # //*** Return r2 score
            return round(r2_score(input_df[actual_col], input_df[predict_col]), 4)

        if action == 'rmse':
            if verbose:
                print(f"rmse Actual: {actual_col} Predict: {predict_col}")

            # //*** Return Root Mean Squared Error
            return round(sqrt(mean_squared_error(input_df[actual_col], input_df[predict_col])), 4)

        if action == 'mae':
            if verbose:
                print(f"rmse Actual: {actual_col} Predict: {predict_col}")

            # //*** Return Mean Absolute Error
            return round(mean_absolute_error(input_df[actual_col], input_df[predict_col]), 4)

        # //**************************************
        # //**************************************
        # //**************************************
        # //**** END action=='Accuracy'
        # //**************************************
        # //**************************************
        # //**************************************

    # //*********************************************************************************************************************
    # //*********************************************************************************************************************
    # //**** Build Technical Indicators
    # //**** Technicals handles the processing and building of the technical indicators
    # //**** Each indicator is assigned a separate sub-function for building. Sub-functions are probably not PEP standard
    # //**** It's structured to limit scope and make the code more readable
    # //*********************************************************************************************************************
    # //*********************************************************************************************************************

    def build_technicals(self,technical_cols,**kwargs):
        
        #//*** Calculate the Exponential Moving Average. 
        #//*** Code acquired from 3rd party
        def calculate_ema(col, **kwargs):
            
            days=2
            label="ema"
            smoothing=2
            decimals = 2
            return_ema = False
            
            for key,value in kwargs.items():
                
                if key == 'days':
                    days=value
                if key == 'label':
                    label=value
                if key=='smoothing':
                    smoothing=value
                if key=="decimals":
                    decimals=value
                if key == "return_ema":
                    return_ema = value
            
            prices = self.df[col].values
            
            #print(len(prices),prices)
            ema = [sum(prices[:days]) / days]
            for price in prices[days:]:
                ema.append((price * (smoothing / (1 + days))) + ema[-1] * (1 - (smoothing / (1 + days))))
            
            #//*** No Rounding, return full float value
            out = (list(np.empty(len(prices)-len(ema))*np.nan) + ema)

            # //*** Replace Nan in this_ema with values from the original column. NaNs Break Stuff
            for x in range(value - 1):
                #print(x, out[x], self.df[col].iloc[x])
                out[x] = self.df[col].iloc[x]

            if decimals > -1:
                out = np.array(out).round(decimals=decimals)
               
            #//*** Return just the ema as a Numpy Array
            if return_ema:
                return np.array(out)
            else:
                #//*** Otherwise add it to the interal df
                self.df[label] = out

        #//*** Simple Moving Average
        def build_sma(col,window,**kwargs):
            
            label = "sma"
            return_sma = False
            center=False
            
            for key,value in kwargs.items():
                if key == "label":
                    label = value
                if key == "return_sma":
                    return_sma = value
                if key == "center":
                    center = value
                    
                    
                    
            out = self.df[col].rolling(window=window,center=center).mean()

            
            if return_sma:
                return out
            else:
                self.df[label] = out
            
        #//*** Build Weighted Moving Average
        #//*** Code acquired From: https://towardsdatascience.com/trading-toolbox-02-wma-ema-62c22205e2a9
        #//*** wma10 = data['Price'].rolling(10).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)wma10.head(20)
        def build_wma(col, value, **kwargs):

            label = "wma"
            return_wma = False
            
            for key,val in kwargs.items():
                if key == "label":
                    label = val
                if key == "return_wma":
                    return_wma = val
                    
                    
            print("Label",label)        
            
            #//*** Build Weights
            weights = np.arange(1,value+1)

            #//*** Generates a Rolling window of Value length
            #//*** Each window is multiplied by the weights      
            out = self.df[col].rolling(window=value).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)

            if return_wma:
                return out
            else:
                self.df[label] = out

        def build_dema(col,**kwargs):
            
            days=2
            label="dema"
            smoothing=2
            decimals = 2
            return_dema = False
            
            for key,value in kwargs.items():
                
                if key == 'days':
                    days=value
                if key == 'label':
                    label=value
                if key=='smoothing':
                    smoothing=value
                if key=="decimals":
                    decimals=value
                if key == "return_dema":
                    return_dema = value
            
            first_ema = calculate_ema(col,days=days,return_ema=True)

            #//*** Need to store the first ema temporarily to the df. Start Here
            temp_ema = calculate_ema(col,label="___",days=value,return_ema=True)

            #//*** Replace NaN from first EMA using values from self.df
            for x in range(value):
                if np.isnan(first_ema[x]):
                    first_ema[x] = self.df[col].iloc[x]
                    temp_ema[x] = self.df[col].iloc[x]

            #//*** Store Temp EMA
            self.df["___"] = temp_ema
            
            out = (2*np.array(first_ema)) - calculate_ema("___",days=value,return_ema=True)
            if return_dema:
                return out
            else:
                self.df[label] = out
            
            del self.df["___"]
            
        def build_inverted_curve(primary,reference):

            #diff = primary - reference
            #print(primary + (diff*2))
            #return primary + (diff*2)
            print(primary)
            df = pd.DataFrame()
            df['primary'] = self.df[primary]
            df['reference'] = reference

            out = []
            for row in df.iterrows():

                #//*** Ignore Nan
                if np.isnan(row[1]['primary']):
                    out.append(np.nan)
                    continue

                if np.isnan(row[1]['reference']):
                    out.append(np.nan)
                    continue

                #//*** Generate Difference
                diff = ((row[1]['primary'] - row[1]['reference']))

                diff = ((row[1]['reference'] - row[1]['primary']))
                #diff *= 2

                #//*** Orig Version, might be something here mathematically/precictively
                #result = row[1]['primary']+diff

                #//*** Combine results.
                result = row[1]['primary']- (diff)

                out.append(result)
            

            return pd.Series(out)



        ######################################################
        ######################################################
        ######################################################
        ######################################################

        verbose = False
        
        for key,value in kwargs.items():
            if key == 'verbose':
                verbose = value

        if verbose:
            print("===================")
            print("Begin Tecnicals")
            print("===================")


        for mc in technical_cols:
            
            attrib=None

            
            #//*** Split the column name. Properly formatted, it becomes action, column value
            for x in enumerate(mc.split("_")):
                if x[0] == 0:
                    action = x[1]
                elif x[0] == 1:
                    col = x[1]
                elif x[0] == 2:
                    value = x[1]
                elif x[0] == 3:
                    attrib = x[1]
                    
                
            #action,col,value = mc.split("_")
            value = int(value)
            
            if verbose:
                print("Action:",action,"Column:",col,"Value:",value,"Attrib:",attrib)
            
            while "/" in col:
                col=col.replace("/","_")
            
            
            #print("Action:",action,"Column:",col,"Value:",value,"Attrib:",attrib)

            #//*** Exponential Moving Average
            if action == 'ema':
                calculate_ema(col,label=mc,days=value)


            #//*** Inverted Exponential Moving Average
            elif action == 'iema':
                this_ema = calculate_ema(col,days=value,return_ema=True)
                #print(this_ema)
                #//*** Replace Nan in this_ema with values from the original column. NaNs Break Stuff
                for x in range(value):
                    #print(x,this_ema[x],self.df[col].iloc[x])
                    this_ema[x] = self.df[col].iloc[x]


                #print("THIS EMA:",this_ema)


                
                temp_ema =  build_inverted_curve(col,this_ema).values

                self.df["___"] = temp_ema

                if attrib == "ionly":
                    #//*** Return just the inverted curve
                    self.df[mc] = self.df["___"]
                else:  
                    #//*** Perform EMA on inverted values
                    self.df[mc] = calculate_ema("___",days=value-1,return_ema=True)


                #self.df[mc] = temp_df['temp']
                         
                   
                #tech_df[mc] = build_inverted_curve(input_df[col],calculate_ema(input_df[col],value)).rolling(window=value).mean()

            #//*** Double Exponential Moving Average
            elif action == 'dema':
                build_dema(col,label=mc,days=value)
            
            elif action == 'idema':

                #//*** Build the DEMA
                this_dema = build_dema(col,days=value, return_dema=True)
                print(mc,col,value)
                print(this_dema)
                #//*** Replace Nan in this_ema with values from the original column. NaNs Break Stuff
                for x in range(value):
                    #print(x,this_dema[x],self.df[col].iloc[x])
                    this_dema[x] = self.df[col].iloc[x]
                
                #//*** Generate the Inverted Values and store temp column
                tds = build_inverted_curve(col,this_dema).values

                print(tds)
                #//*** Reset the first elements to the original values. NaNs mess up the math
                for x in range(value):
                    tds[x] = self.df[col].iloc[x]
                self.df["___"] = tds
                #//*** Return just the inverted curve
                self.df[mc] = build_dema("___",days=value, return_dema=True)

            #//*** Simple moving Average
            elif action == 'sma':
                build_sma(col,value,label=mc,center=False)

            #//*** Inverted Simple moving Average
            elif action == 'isma':
                
                #//*** Build the SMA
                this_sma = build_sma(col,value,return_sma=True)
                
                #//*** Generate the Inverted Values and store temp column
                self.df["___"] = build_inverted_curve(col,this_sma) 
                
                if attrib == "ionly":
                    #//*** Return just the inverted curve
                    self.df[mc] = self.df["___"]
                else:  
                    #//*** Perform SMA on inverted values
                    self.build_sma("___",value,label=mc)
                
                #//*** Delete Temp Column
                del self.df["___"]

            elif action == 'cma':
                build_sma(col,value,label=mc,center=True)

            elif action == 'wma':
                build_wma(col,value,label=mc)      
                
            elif action == 'iwma':
                
                this_wma = build_wma(col,value, return_wma=True)
                
                self.df["___"] = build_inverted_curve(col, this_wma)
                
                if attrib == "ionly":
                    self.df[mc] = self.df["___"]
                else:
                    self.df[mc] = build_wma("___",value,label=mc, return_wma=True)
                #temp_df = pd.DataFrame()
                #temp_df['temp'] =  build_inverted_curve(input_df[col]) 
                #tech_df[mc] = build_wma(temp_df,'temp',value)



            else:
                print(f"Unknown Value in Column Name: {model} : {model_cols}\nAction: {action}")
                break
            
            #//***Delete Temp Column if it exists
            if "___" in self.df.columns:
                del self.df["___"]
        

        #//**** Check for all NaN Columns
        for col in self.df.columns:
            #//*** A column is all NaN. This will need to be fixed
            if self.df[col].count() == 0:
                print("A Column is filled with NaN. This is a problem and needs to be fixed!!!")
                print(self.df)
                return

        #//*** Drop NaN values at the beginning of the dataset
        self.df.dropna(inplace=True)
        return




    def derived_stats(self, **kwargs):
        print("Derived Stats")
        action = None
        col = None
        label = None

        window = 3
        highest_corr = False

        for key,value in kwargs.items():

            if key == 'col':
                col  = value

            if key == 'action':
                action = value

            if key == 'window':
                window = value

            if key == 'highest_corr':
                highest_corr = value

            if key == 'label':
                label = value
        #//*** Label defaults to action value if unassigned.
        if label == None:
            label = action

        print("Action:",action, "Col:",col, "Label:",label)

        if action == "diff":
            if col == None:
                print("Column required: col=col_name")
                return
            # //*** Generate Daily Difference values
            vals = self.df[col].diff().values
            vals[0] = 0
            self.df[label] = vals
            return

        if action == "ssa":

            if col == None:
                print("Column required: col=col_name")
                return

            transformer = SingularSpectrumAnalysis(window_size=window)
            ssa_vals = transformer.transform(self.df[col].values.reshape(1, -1))

            #//*** Return SSA column with the highest absolute correlation
            if highest_corr:

                #//** initialize Temprary Dataframe
                tdf = pd.DataFrame()

                #//*** Add Target Column
                tdf[col] = self.df[col]

                #//*** add each ssa column to the temp df
                for x in range(1,window):

                    #//*** Assign the first SSA Val
                    tdf[x] = ssa_vals[x]

                #//*** Generate Correlations. Return Target Column and replace 1 with 0
                tdf = abs(tdf.corr().loc[col]).replace(1,0)

                #//*** Return the index value of the SSA with the highest correlation
                tgt_window = tdf[tdf == tdf.max()].index.values[0]
                self.df[label] = ssa_vals[tgt_window]
            else:
                #//*** Return last SSA column
                #for x in range(1,window):

                    #//*** Assign the first SSA Val
                 #   self.df[f"{label}-{x}"] = ssa_vals[x]
                self.df[label] = ssa_vals[window-1]

            return    

#//*** alpha_vantage class is used to download stock prices from alphavantage.co.
#//*** A valid API key is required when initializing the class.
class alpha_vantage():
    def __init__(self,**kwargs):


        self.av_apikey = ""

        for key,value in kwargs.items():
            if key == 'api_key':
                self.av_apikey = value

        pass
    #//******************************************************************************
    #//*** Builds the URL request based on the symbol and type of data requested.
    #//*** Initially, this does the daily numbers.
    #//*** Can easily be scaled up to add many different URL request types
    #//******************************************************************************
    def build_url(self,input_action,input_symbol,m=1,y=1):

        av_apikey = self.av_apikey

        #//*** Valid Actions:
        #//*******  Daily: Gets the historical daily closing price for up to 20 years

        if input_action == 'daily':
            action = "TIME_SERIES_DAILY"
            out = ""
            out += f'https://www.alphavantage.co/query?'
            out += f'function={action}'
            out += f'&symbol={input_symbol}'
            out += f'&outputsize=full'
            out += f'&apikey={av_apikey}'

            return out

        if input_action == '1min':
            action = "TIME_SERIES_INTRADAY_EXTENDED"
            out = ""
            out += f'https://www.alphavantage.co/query?'
            out += f'function={action}'
            out += f'&symbol={input_symbol}'
            out += f'&outputsize=full'
            out += f'&slice=year{y}month{m}'
            out += f'&interval=1min'
            #out += "datatype=json"
            #out += f'&adjusted=true',
            #out += "&slice=year1month1",
            out += f'&apikey={av_apikey}'

            return out

        if input_action == '60min':
            action = "TIME_SERIES_INTRADAY_EXTENDED"
            out = ""
            out += f'https://www.alphavantage.co/query?'
            out += f'function={action}'
            out += f'&symbol={input_symbol}'
            out += f'&outputsize=full'
            out += f'&slice=year{y}month{m}'
            out += f'&interval=60min'
            #out += "datatype=json"
            #out += f'&adjusted=true',
            #out += "&slice=year1month1",
            out += f'&apikey={av_apikey}'

        return out
        print(f"Invalid Action: {input_action}")
        print(f"No URL Returned, PLease try again")
        return None



    def get_stocks(self,**kwargs):
        from pathlib import Path
        import os
        import pandas as pd
        import requests
        import time

        #action = '1min'
        #action = '60min'
        action = 'daily'
        av_apikey = None
        symbols = []
        save_files = True
        save_folder = "stocks"
        wait_length = 20
        for key,value in kwargs.items():
            av_apikey = self.av_apikey

            if key == 'symbols':
                symbols = value

            if key == 'save_folder':
                save_folder = value

            if key == 'wait_length':
                wait_length = value


        #//*** Get Working Directory
        current_dir = Path(os.getcwd()).absolute()

        

        save_dir = current_dir.joinpath(save_folder)

        #//*** Create the Save Folder if it doesn't exist
        if not save_dir.is_dir():
            os.mkdir(save_dir)
            print("Exists:", save_dir.is_dir())

        for symbol in symbols:
            print("Downloading:", symbol, "at", action, "intervals")

            if (action == '1min') or (action == '60min'):

                #//*** initialize output dataframe
                out_df = pd.DataFrame()

                #//*** Loop the year
                for year in [1,2]:

                    #//*** Loop each month
                    for month in range(1,13):
                        print(f"Length out_df: {len(out_df)}")
                        print(f"Building URL: {symbol} - Month {month} Year {year}")
                        url = build_url(action,symbol,month,year)

                        print("Downloading")
                        print(url)
                        r = requests.get(url)
                        print(r.text[:1000])
                        f = open("t.csv", "w")

                        f.write(r.text)
                        f.close()

                        out_df = pd.concat([out_df,pd.read_csv("t.csv")])

                        print("Waiting 20 Seconds")
                        time.sleep(20)


                print("df Built")
                print(out_df.head(10))

                output_filename = f"./stocks/{symbol}_{action}.csv.zip"

                #//*** Convert Path to Mac formatting if needed
                #if platform.system() == 'Darwin':
                #output_filename = output_filename.replace("\\","/")

                print(f"Writing dataframe to File: {output_filename}")
                out_df.to_csv(output_filename,compression="zip",index=False)
            if action == 'daily':

                #//*** initialize output dataframe
                out_df = pd.DataFrame()

                print(f"Building URL: {symbol}")
                url = self.build_url(action,symbol)

                print("Downloading")
                #print(url)

                r = requests.get(url)


                #//*** Convert raw string to dictionary for processing
                data = r.json()
                try:
                    #//*** Get the Key value that contains the list of dates
                    data_key = list(data.keys())[1]
                except:
                    print("Problem wth data_key:")
                    #print("data_key:",data_key)
                    print("data:",data)
                    print("Skipping Symbol:",symbol)
                    continue

                #//*** Output Dictionary
                out_dict = {}
                print("Processing....")
                #//*** Process Data into the out_dict
                for date in data[data_key]:
                    #//*** Build out_dict (output_dictionary) keys
                    if len(out_dict.keys()) == 0:
                        out_dict['date'] = []
                        out_dict['symbol'] = []

                        #//*** Get this dictionary for the first row. Use the key values, but strip the first 3 characters which are numeric
                        for key in data[data_key][date].keys():
                            out_dict[key[3:]] = []

                    #//*** Add Date to out_dict
                    out_dict['date'].append(date)

                    #//*** Add Symbol to out_dict
                    out_dict['symbol'].append(symbol)

                    #//*** Loop through the daily values and append to the out_dict
                    for key,value in data[data_key][date].items():

                        #//*** Trim first 3 characters off key and append to the appropriate dictionary list
                        out_dict[key[3:]].append(value)

                print("Building Dataframe")
                out_df = pd.DataFrame()
                #//*** Convert the Dictionary to a Dataframe
                #//*** Each Key is a column, the data is the list
                for key,value in out_dict.items():
                    out_df[key] = value

                #//*** Generic Filename - Placeholder
                output_filename = f"{symbol}_need_a_better_name.csv.zip"


                #//*** Build filename based on action type
                if action == 'daily':
                    output_filename = f"{symbol}_daily.csv.zip"

                if action == '1min':
                    output_filename = f"{symbol}_1min.csv.zip"

                output_filepath = save_dir.joinpath(output_filename)

                print(f"Writing dataframe to File: {output_filename}")
                out_df.to_csv(output_filepath,compression="zip",index=False)

                #else:
            #    print("We've got an url problem Skipping")

            #//*** Wait 20 seconds so we don't hammer the API
            #//*** Max is 5 calls / minute & 500 /day

            print(f"Waiting {wait_length} Seconds")
            time.sleep(wait_length)

    






