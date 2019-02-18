"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
            
             _   _        _____ _              ______             _ _      _
            | \ | |      /  ___| |             | ___ \           | (_)    | |
            |  \| |______\ `--.| |_ ___ _ __   | |_/ / __ ___  __| |_  ___| |_ ___  _ __
            | . ` |______|`--. \ __/ _ \ '_ \  |  __/ '__/ _ \/ _` | |/ __| __/ _ \| '__|
            | |\  |      /\__/ / ||  __/ |_) | | |  | | |  __/ (_| | | (__| || (_) | |
            \_| \_/      \____/ \__\___| .__/  \_|  |_|  \___|\__,_|_|\___|\__\___/|_|
                                       | |
                                       |_|
                                                                    twitter: ihsnsulaiman
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from tkinter import scrolledtext, messagebox, filedialog
from os import path
import math
from tkinter import *
from tkinter.ttk import *
import tkinter.font

# set seed
np.random.seed(7)

# initial the variables
data = None
train_train = []
train_test = []
train = []
x_train = []
lags = 1
lags_value = []
step_number = 0
first_value = -1
UI = None
mdl = None
min_neuron = 1
max_neuron = 5
last_data_index = []

# initialize the fields check arrays
model_is_ready_to_train = [False, False, False]
model_is_ready_to_predict = [False, False, False]


class AppUI(Tk):
    def __init__(self):
        Tk.__init__(self)

        dFont = tkinter.font.Font(family="Arial", size=8)

        # The first row
        self.fm1 = Frame(self, relief=RAISED, borderwidth=1)
        self.dataset_lbl = Label(self.fm1, text="Dataset : ", font=("Arial Bold", 8))
        self.dataset_lbl.pack(side=LEFT, padx=5, pady=5)
        self.path_btn = Button(self.fm1, text="Open", command=get_data_set)
        self.path_btn.pack(side=RIGHT, padx=5, pady=5)
        self.fm1.pack(fill=BOTH, side=TOP, expand=True)

        # The second row
        self.fm2 = Frame(self, relief=RAISED, borderwidth=1)
        self.min_nrn_lbl = Label(self.fm2, text="Min Neuron : ", font=("Arial Bold", 8))
        self.min_nrn_lbl.pack(side=LEFT, padx=5, pady=5)
        self.min_nrn_entry = Entry(self.fm2, width=5)
        self.min_nrn_entry.pack(side=LEFT, padx=5, pady=5)
        self.max_nrn_lbl = Label(self.fm2, text="Max Neuron : ", font=("Arial Bold", 8))
        self.max_nrn_lbl.pack(side=LEFT, padx=5, pady=5)
        self.max_nrn_entry = Entry(self.fm2, width=5)
        self.max_nrn_entry.pack(side=LEFT, padx=5, pady=5)
        self.set_neurons_btn = Button(self.fm2, text="set", command=self.set_neurons)
        self.set_neurons_btn.pack(side=RIGHT, padx=5, pady=5)
        self.fm2.pack(fill=BOTH, side=TOP, expand=True)

        # The third row
        self.fm3 = Frame(self, relief=RAISED, borderwidth=1)
        self.lags_lbl = Label(self.fm3, text="Lags : ", font=("Arial Bold", 8))
        self.lags_lbl.pack(side=LEFT, padx=5, pady=5)
        self.set_lags_btn = Button(self.fm3, text="set", command=self.set_lags)
        self.set_lags_btn.pack(side=RIGHT, padx=5, pady=5)
        self.lags_entry = Entry(self.fm3, width=15)
        self.lags_entry.insert(END, '1,2,3')
        self.lags_entry.pack(side=RIGHT, padx=5, pady=5)
        self.lags_progress_txt = Label(self.fm3, text="", font=("Arial Bold", 9))
        self.lags_progress_txt.pack(side=RIGHT, padx=5, pady=5)
        self.fm3.pack(fill=BOTH, side=TOP, expand=True)

        # The fourth row
        self.fm4 = Frame(self, relief=RAISED, borderwidth=1)
        self.progress = Progressbar(self.fm4, orient='horizontal', length=200, mode='determinate')
        self.progress.pack(side=LEFT, fill=X, padx=3, pady=3)
        self.train_btn = Button(self.fm4, text="Train", command=check_train_fields)
        self.train_btn.pack(side=RIGHT, padx=5, pady=5)
        self.fm4.pack(fill=BOTH, side=TOP, expand=True)

        # The fifth row
        self.fm5 = Frame(self, relief=RAISED, borderwidth=1)
        self.step_num_lbl = Label(self.fm5, text="how many step do you want to predict : ", font=("Arial Bold", 8))
        self.step_num_lbl.pack(side=LEFT, padx=5, pady=5)
        self.set_step_btn = Button(self.fm5, text="set", command=self.set_steps)
        self.set_step_btn.pack(side=RIGHT, padx=5, pady=5)
        self.step_num_entry = Entry(self.fm5, width=5)
        self.step_num_entry.pack(side=RIGHT, padx=5, pady=5)
        self.fm5.pack(fill=BOTH, side=TOP, expand=True)

        # The sixth row
        self.fm6 = Frame(self, relief=RAISED, borderwidth=1)
        self.first_value_lbl = Label(self.fm6, text="Your first value : ", font=("Arial Bold", 8))
        self.first_value_lbl.pack(side=LEFT, padx=5, pady=5)
        self.first_value_btn = Button(self.fm6, text="set", command=self.set_first_value)
        self.first_value_btn.pack(side=RIGHT, padx=5, pady=5)
        self.first_value_entry = Entry(self.fm6, width=5)
        self.first_value_entry.pack(side=RIGHT, padx=5, pady=5)
        self.fm6.pack(fill=BOTH, side=TOP, expand=True)

        # The seventh row
        self.fm7 = Frame(self, relief=RAISED, borderwidth=1)
        self.predict_btn = Button(self.fm7, text="Predict", command=check_predict_fields)
        self.predict_btn.pack(side=RIGHT, padx=5, pady=5)
        self.fm7.pack(fill=BOTH, side=TOP, expand=True)

        # The eighth row
        self.fm8 = Frame(self, relief=RAISED, borderwidth=1)
        self.results_dump = scrolledtext.ScrolledText(self.fm8, width=30, height=15, font=dFont)
        self.results_dump.pack(fill=BOTH, padx=3, pady=3)
        self.fm8.pack(fill=BOTH, side=TOP, expand=True)

    global model_is_ready_to_train

    # the neurons setting function
    def set_neurons(self):
        global min_neuron, max_neuron
        minstr = self.min_nrn_entry.get()
        maxstr = self.max_nrn_entry.get()
        min_neuron = int(minstr) if isitnumber(minstr) is True else 0
        max_neuron = int(maxstr) if isitnumber(maxstr) is True else 0

        if min_neuron >= max_neuron:
            messagebox.showinfo('Error: Wrong Value',
                                "- the values must be a number \n"
                                "- minimum value can't be equal or bigger then maximum value")
        else:
            model_is_ready_to_train[1] = True

    # FUNCTION FOR LETTING THE USER SET MORE THEN ONE LAG
    def set_lags(self):

        global lags, lags_value
        lags_value = []
        model_is_ready_to_train[2] = True
        lags_tmp = self.lags_entry.get()
        if (lags_tmp[0] == ",") or (lags_tmp[len(lags_tmp)-1] == ","):
            model_is_ready_to_train[2] = False
        lags = 0
        for x in lags_tmp.split(","):
            if x.isdigit():
                lags_value.append(int(x))
                lags += 1
            else:
                model_is_ready_to_train[2] = False
                messagebox.showinfo('Error: Wrong Value',
                                    "- please enter correct lags value\n"
                                    "  Ex: 2,5,7,10")
                break

        if lags_value is []:
            model_is_ready_to_train[2] = False

        if data is not None:
            acl_data, _ = prepare_data(train, lags_value)
            self.lags_progress_txt.configure(text="Autocorrelation is {}".format(round(acl_func(acl_data, lags), 3)))
            self.update()

    def set_steps(self):
        global step_number, model_is_ready_to_predict
        steptmp = self.step_num_entry.get()
        step_number = int(steptmp) if isitnumber(steptmp) is True else None
        step_number += 1

        if step_number:
            model_is_ready_to_predict[1] = True

    def set_first_value(self):
        global first_value, model_is_ready_to_predict
        frsttmp = self.first_value_entry.get()
        first_value = int(frsttmp) if isitnumber(frsttmp) is True else None
        if first_value:
            model_is_ready_to_predict[2] = True


def get_data_set():
    global train_train, train_test, train, data, data_length, model_is_ready_to_train
    data_set_path = filedialog.askopenfilename(initialdir=path.dirname(__file__))

    if data_set_path is not '':
        # import data set
        df = pd.read_csv(data_set_path, sep=';', parse_dates=True, index_col=0)
        # replace the nan values with 0
        df = df.replace(np.nan, 0)

        data = df.values

        # using keras often requires the data type float32
        data = data.astype('float32')

        data_length = len(data)

        # slice the data
        train_train = data[0:(data_length*80//100), :]                                                # 80%
        train_test = data[(data_length*80//100):data_length, :]                                       # 20%

        train = data[:, :]                                                                            # 100%

        if data is not None:
            model_is_ready_to_train[0] = True


def isitnumber(usr_input):
    return usr_input.replace('.', '', 1).isdigit()


def prepare_data(data, lags_value):
    """
    Create lagged data from an input time series
    """
    global last_data_index
    x, y = [], []
    for row in range(len(data) - lags_value[-1]):
        a = []
        for lags_row in lags_value:
            a.append(data[lags_row-1, 0])
        x.append(a)
        y.append(data[lags_value[-1], 0])

        lags_value = [t+1 for t in lags_value]

    last_data_index = [t-2 for t in lags_value]

    return np.array(x), np.array(y)


def acl_func(acl_data, lags):
    acl_data = acl_data.flatten()
    mean = np.mean(acl_data)
    numerator = 0
    denominator = 0
    for i in range(0, len(acl_data)-lags-1):
        numerator += (acl_data[i] - mean)*(acl_data[i+lags] - mean)
    for i in range(0, len(acl_data)-1):
        denominator += (acl_data[i] - mean)**2
    return numerator/denominator


def check_train_fields():
    """ check if the the train fields is ready """

    if [i for i in model_is_ready_to_train if i is False]:
        messagebox.showinfo('Error: Missing Value', 'All field must be set. Please check it  and try again ')
    else:
        try:
            train_function()
        except :
            messagebox.showinfo('Logical Error',
                                'please choose a suitable lags values for your dataset and try again.')


def train_function():
    global model_is_ready_to_predict, mdl, UI, x_train

    # prepare the data
    x_train, y_train = prepare_data(train_train, lags_value)
    x_test, y_test = prepare_data(train_test, lags_value)

    UI.progress["maximum"] = 100
    best_layer_number = -1
    best_test_score = 9999999

    counter = min_neuron  # initialize counter
    progress_value = 0

    while counter <= (max_neuron + 1):
        # progress bar control
        progress_value += 100 / (2 * ((max_neuron - min_neuron) + 2))
        UI.progress["value"] = progress_value
        UI.update()

        # create the model
        mdl = Sequential()

        # prepare for the main model
        if counter == (max_neuron + 1):
            x_train, y_train = prepare_data(train, lags_value)
            mdl.add(Dense(best_layer_number, input_dim=lags, activation='relu'))
        else:
            mdl.add(Dense(counter, input_dim=lags, activation='relu'))

        mdl.add(Dense(1))
        mdl.compile(loss='mean_squared_error', optimizer='adam')

        # start the training
        mdl.fit(x_train, y_train, epochs=100, batch_size=1, verbose=2)

        # estimate model performance
        train_score = mdl.evaluate(x_train, y_train, verbose=0)
        test_score = mdl.evaluate(x_test, y_test, verbose=0)

        # progress bar control
        progress_value += 100 / (2 * ((max_neuron - min_neuron) + 2))
        UI.progress["value"] = progress_value
        UI.update()

        # find the minimum Test Score with RMSE unit
        if best_test_score > test_score and counter < max_neuron:
            best_test_score = test_score
            best_layer_number = counter
        elif counter is max_neuron:
            # save the model
            mdl.save('best_model.h5')

        counter += 1

    model_is_ready_to_predict[0] = True

    # prepare the model summary string
    summary = []
    mdl.summary(print_fn=lambda x: summary.append(x))
    short_model_summary = '\n'.join(summary)
    mdl.summary()

    # print the model information
    UI.results_dump.insert(INSERT, "The Best Neuron number is: {}\n".format(best_layer_number))
    UI.results_dump.insert(INSERT, 'The Best Test Score: {:.2f} MSE ({:.2f} RMSE)'
                           .format(test_score, math.sqrt(test_score)) + "\n")
    UI.results_dump.insert(INSERT, "Model Summary: \n{}".format(short_model_summary) + "\n")


def check_predict_fields():
    """ check if the model is finished the train and the predict fields is ready """

    if [i for i in model_is_ready_to_predict if i is False]:
        messagebox.showinfo('Error: Missing Value', 'All field must be set. Please check it  and try again ')
    else:
        predict_function()


def predict_function():
    global first_value
    predict_array, f = [], []
    f = x_train[-1]
    for i in range(step_number):
        f = f[1:]
        f = np.append(f, first_value)
        predict_array.append(f)
        last_predicted_array = mdl.predict(np.array(predict_array))
        UI.results_dump.insert(INSERT, str(i) + " : " + str(last_predicted_array[-1]) + "\n")
        first_value = last_predicted_array[-1]


def main():
    global UI
    UI = AppUI()
    UI.option_add('*font', ('verdana', 12, 'bold'))
    UI.title("N-step Predictor")
    UI.mainloop()


if __name__ == '__main__':
    main()
