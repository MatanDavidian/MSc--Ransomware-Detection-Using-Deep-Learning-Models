import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, LSTM, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder
from benign_train_test_split_options import options
from sklearn.utils import resample
import gc
import re
import fasttext.util
import os
import time
from keras import backend as K
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

Epochs = 30
ft = fasttext.load_model('cc.en.300.bin')
logLevel = 1
bert_encoding_cache = {}
print("TensorFlow version:", tf.__version__)
print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)

# Create a dictionary to store results
results = {
    'amoutOfSystemcalls': [],
    'skip': [],
    'Features': [],
    'Model': [],
    'Train_Accuracy': [],
    'Test_Accuracy': [],
    'Sensitivity': [],
    'Specificity': [],
    'F1_Score': [],
    'Train_Time': [],
    'Test_Time': [],
    'Train_Set_Shape': [],
    'Test_Set_Shape': [],
    'TP': [],  # True Positives
    'TN': [],  # True Negatives
    'FP': [],  # False Positives
    'FN': [],  # False Negatives
    'encoding': [],
    'win_size': [],
    'op': []
}


def make_windows_per_process(data, labels, process_name, window_size, step, fixed_window_count, encoding):
    X = []
    Y = []
    for process in process_name.unique():
        process_data = data[process_name == process]
        process_data = np.array(process_data.drop(columns=['Process Name']))
        process_labels = labels[process_name == process]
        windows = []
        for i in range(0, len(process_data) - window_size, step):
            window = process_data[i:i + window_size]
            windows.append(window)

        window_count = len(windows)

        # If the process has fewer windows, create more by oversampling
        while window_count < fixed_window_count:
            # Randomly select a window to duplicate
            window_to_duplicate = resample(windows, replace=False, n_samples=1)[0]
            # Duplicate the window and its corresponding label
            windows.append(window_to_duplicate)
            window_count += 1

        X.extend(windows)
        # Assume all windows from the same process have the same label
        Y.extend([process_labels[0]] * window_count)
        if logLevel == 2:
            print(f"Process: {process}, Class: {process_labels[0]}, Windows: {window_count}")

    if encoding == 'FT':
        res = list_to_4d_numpy(X)
    elif encoding == 'BERT':
        res = list_to_4d_numpy_bert(X)
    else:
        res = np.array(X)

    return res, np.array(Y)


def list_to_4d_numpy_bert(data):
    num_rows = len(data)
    print(f"Starting conversion of data with {num_rows} rows to 4D numpy array.")

    # Get the shape of the 2D numpy array inside the first element of the list
    sample_element = data[0]
    element_rows, element_cols = sample_element.shape
    print(f"Each data element has shape: {element_rows} x {element_cols}")

    data_4d = np.zeros((num_rows, element_rows, element_cols, 768))

    for i in range(num_rows):
        for j in range(element_rows):
            for k in range(element_cols):
                data_4d[i, j, k] = data[i][j][k]

    print(f"Converted data to 4D numpy array with shape: {data_4d.shape}")
    return data_4d


def list_to_4d_numpy(data):
    num_rows = len(data)

    # Get the shape of the 2D numpy array inside the first element of the list
    sample_element = data[0]
    element_rows, element_cols = sample_element.shape

    data_4d = np.zeros((num_rows, element_rows, element_cols, 300))

    for i in range(num_rows):
        for j in range(element_rows):
            for k in range(element_cols):
                data_4d[i, j, k] = data[i][j][k]

    return data_4d


def separate_detail_column_old(df):
    # Extract attributes using regex
    attributes_pattern = r'(?P<attribute>\w+):\s(?P<value>[^,]+)'

    # Function to extract the attributes
    def extract_attributes(detail_string):
        matches = re.findall(attributes_pattern, detail_string)
        return dict(matches)

    # Apply function to the 'Detail' column
    df_details = df['Detail'].fillna('').apply(extract_attributes).apply(lambda x: pd.Series(x, dtype='object'))

    # Concatenate original dataframe and extracted details
    result_df = pd.concat([df, df_details], axis=1)

    # Drop the original 'Detail' column
    result_df.drop(columns=['Detail'], inplace=True)

    return result_df


def separate_detail_column(df):
    # Attributes that you want to extract
    details_taken_column = ["FileAttributes:", "DeletePending:"
                                               "Disposition:", "Options:", "Attributes:", "ShareMode:",
                            "Access:", "Exclusive:",
                            "FailImmediately:", "OpenResult:", "PageProtection:", "Control:",
                            "ExitStatus:", "PrivateBytes:", "PeakPrivateBytes:", "WorkingSet:",
                            "PeakWorkingSet:", "Commandline:"
                                               "Priority:", "GrantedAccess:", "Name:", "Type:", "Data:", "Query:",
                            "HandleTags:",
                            "I/OFlags:",
                            "FileSystemAttributes:", "DesiredAccess:"]
    # "Filter:", , "Offset:" , "FileInformationClass:" , "SyncType:" , "Length:",
    # "Index:", "KeySetInformationClass:",  "MaximumComponentNameLength:", "FileSystemName:", , "0:00", "1:00", "2:00"
    # "Directory:", "IndexNumber:", "ImageSize:", "ImageBase:", "AllocationSize:",
    # "EndOfFile:", "NumberOfLinks:" "Currentdirectory:",

    # "Index:", "SubKeys:", "KeySetInformationClass:",  "MaximumComponentNameLength:", "FileSystemName:",
    # "FileSystemAttributes:", "Impersonating:", "Values:", , "DesiredAccess:",

    # details_taken_column = [item.strip() for item in ' '.join(details_taken_column).split(':') if item.strip()] + [':']

    # Extract attributes using regex
    attributes_pattern = r'(?P<attribute>\w+):\s(?P<value>[^,]+)'

    # Function to extract the attributes
    def extract_attributes(detail_string):
        matches = re.findall(attributes_pattern, detail_string)
        return {k: v for k, v in matches if k + ':' in details_taken_column}

    # Apply function to the 'Detail' column
    try:
        df['Detail'] = df['Detail'].fillna('')  # Fill NA values with empty string
        df_details = df['Detail'].apply(extract_attributes).apply(pd.Series)
    except Exception as e:
        print(f"Error processing Detail column: {e}")
        return df  # Return original DataFrame in case of error

    # Concatenate original dataframe and extracted details
    result_df = pd.concat([df, df_details], axis=1)

    # Drop the original 'Detail' column
    result_df.drop(columns=['Detail'], inplace=True)

    return result_df


def norm_data(df, encoding):
    print("norm_data")
    numeric_c = []
    if encoding == 'FT':
        dimension = 300
        t_empty_list = np.zeros(300, dtype=np.float32)
    elif encoding == 'BERT':
        dimension = 768
        t_empty_list = np.zeros(768, dtype=np.float32)
    elif encoding == 'OH':
        return df, numeric_c
    else:
        raise Exception("unknown encoding")

    for c in df.columns:
        if c == 'Ransomware':
            continue

        # Replace 'nan' and 'n/a' with np.nan
        df[c].replace(['nan', 'n/a'], np.nan, regex=True, inplace=True)

        # Try to convert column to numeric
        try:
            df[c] = pd.to_numeric(df[c])
            if df[c].max() - df[c].min() != 0:  # Avoid division by zero
                min_c = df[c].min()
                df[c] = (df[c] - min_c) / (df[c].max() - min_c)
                numeric_c.append(c)

                df[c] = df[c].apply(lambda x: process_value(x, dimension))
                # df[c] = df[c].apply(lambda x: np.array([x] + np.zeros(299, dtype=np.float32)))

                # print("cast to numeric:")
                # print(c)
        except:
            pass

    # Replace NaN values with ndarray of zeros
    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: t_empty_list if isinstance(x, (list, np.ndarray)) and pd.isnull(x).any() else x)
    print("numeric cols:")
    print(numeric_c)
    print("end normalized")
    return df, numeric_c


def process_value(x, dimension):
    if isinstance(x, (int, float, np.number)):
        return np.array([x] + list(np.zeros(dimension - 1, dtype=np.float32)))
    else:
        return np.zeros(dimension, dtype=np.float32)


# Function to load data
def load_data(virus_path, benign_path, train_ids_file, test_ids_file, num_system_calls, features, op=0):
    print("Loading benign data...")
    benign_files = [f for f in os.listdir(benign_path) if f.endswith('.csv')]
    benign_dfs = [pd.read_csv(os.path.join(benign_path, f), usecols=['Process Name'] + features) for f in benign_files]
    benign_df = pd.concat(benign_dfs)

    benign_df['Ransomware'] = 0

    grouped = benign_df.groupby('Process Name')
    benign_dfs = []
    for name, group in grouped:
        group = group.iloc[:num_system_calls]
        benign_dfs.append(group)
    benign_df = pd.concat(benign_dfs)

    process_names = benign_df['Process Name'].unique()
    # train_processes = np.random.choice(process_names, size=int(len(process_names) * 0.8), replace=False)
    train_processes = options[op]["train_processes"]
    print(f"train_processes: {train_processes}")
    # test_processes = list(set(process_names) - set(train_processes))
    test_processes = options[op]["test_processes"]
    print(f"test_processes: {test_processes}")
    benign_train_df = benign_df[benign_df['Process Name'].isin(train_processes)]
    benign_test_df = benign_df[benign_df['Process Name'].isin(test_processes)]
    if logLevel > 1:
        print("Benign train class distribution:", np.unique(benign_train_df['Ransomware'], return_counts=True))
        print("Benign test class distribution:", np.unique(benign_test_df['Ransomware'], return_counts=True))

    print("Loading virus data...")
    with open(os.path.join(virus_path, train_ids_file), "r") as file:
        train_ids = file.readlines()
    with open(os.path.join(virus_path, test_ids_file), "r") as file:
        test_ids = file.readlines()
    train_ids = ["VirusShare_" + id.strip() + ".csv" for id in train_ids]
    test_ids = ["VirusShare_" + id.strip() + ".csv" for id in test_ids]

    train_virus_dfs = []
    test_virus_dfs = []
    if logLevel > 1:
        print("train_ids" + train_ids.__str__())
        print("test_ids" + test_ids.__str__())
    for f in train_ids:
        file_path = os.path.join(virus_path, f)
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            df = pd.read_csv(file_path, nrows=50000, usecols=['Process Name'] + features)
            filtered_df = df[df['Process Name'].str.contains(f[:-4])]
            if not filtered_df.empty:
                filtered_df = filtered_df.iloc[:num_system_calls]
                train_virus_dfs.append(filtered_df)
            else:
                print(f"Filtered DataFrame for {f} is empty.")
        else:
            print(f"{f} does not exist or is empty.")

    for f in test_ids:
        file_path = os.path.join(virus_path, f)
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            df = pd.read_csv(file_path, nrows=50000, usecols=['Process Name'] + features)
            filtered_df = df[df['Process Name'].str.contains(f[:-4])]
            if not filtered_df.empty:
                filtered_df = filtered_df.iloc[:num_system_calls]
                test_virus_dfs.append(filtered_df)
            else:
                print(f"Filtered DataFrame for {f} is empty.")
        else:
            print(f"{f} does not exist or is empty.")
    if logLevel > 1:
        print("train_virus_dfs[0] head:\n", train_virus_dfs[0].head())

    train_virus_df = pd.concat(train_virus_dfs)
    test_virus_df = pd.concat(test_virus_dfs)
    if logLevel > 1:
        print("train_virus_df head:\n", train_virus_df.head())
        print("train_virus_df tail:\n", train_virus_df.tail())

    train_virus_df['Ransomware'] = 1
    test_virus_df['Ransomware'] = 1
    if logLevel > 1:
        print("Virus train class distribution:", np.unique(train_virus_df['Ransomware'], return_counts=True))
        print("Virus test class distribution:", np.unique(test_virus_df['Ransomware'], return_counts=True))

    train_df = pd.concat([benign_train_df, train_virus_df])
    test_df = pd.concat([benign_test_df, test_virus_df])
    if logLevel > 1:
        print("Overall train class distribution:", np.unique(train_df['Ransomware'], return_counts=True))
        print("Overall test class distribution:", np.unique(test_df['Ransomware'], return_counts=True))

    print("Data loading complete.")
    return train_df, test_df


def zero_padding(df, win_size, features, target='Ransomware'):
    print("start zero_padding")
    df = df.set_index(['Process Name', df.groupby('Process Name').cumcount()])
    indexes = []
    for name, group in df.groupby('Process Name'):
        pad = win_size - (len(group) % win_size)
        if pad == win_size:
            pad = 0
        indexes.append(len(group) + pad)
        print(f"name: {name}, pad with: {pad}")

    names = []
    for i in range(len(indexes)):
        name = [df.index.levels[0][i]] * indexes[i]
        names += name
    indexes = [range(i) for i in indexes]
    indexes = [list(i) for i in indexes]
    # flatten
    indexes = [item for sublist in indexes for item in sublist]

    arr = [names, indexes]
    mux = pd.MultiIndex.from_arrays(arr, names=df.index.names)
    df = df.reindex(mux, fill_value=np.nan).reset_index(level=1, drop=True).reset_index()

    # Fill the NaN values with 0
    df[features] = df[features].fillna(0)
    df[target] = df[target].fillna(0).astype(int)

    print("end zero_padding")
    return df


# Function to manually perform one hot encoding for each unique value across both datasets
def manual_one_hot_encode_combined(df, column, unique_values):
    # Create new columns for each unique value
    for value in unique_values:
        df[f"{column}_{value}"] = (df[column] == value).astype(int)

    # Don't drop the original column
    # df = df.drop(column, axis=1)

    return df


def train_model(model, X_train, y_train):
    print("start train")
    start_train_time = time.time()
    try:
        if isinstance(model, sklearn.neural_network.MLPClassifier):
            model.fit(X_train, y_train)
            # For MLPClassifier, directly retrieve the training score after fitting
            training_accuracy = model.score(X_train, y_train)
        else:
            history = model.fit(X_train, y_train, epochs=Epochs, batch_size=16, verbose=1)
            training_accuracy = history.history['accuracy'][0]
    except Exception as e:
        print(e)
        history = model.fit(X_train, y_train, epochs=Epochs, batch_size=16, verbose=1)
        training_accuracy = history.history['accuracy'][0]
    end_train_time = time.time()
    accuracy_train = training_accuracy
    print("accuracy_train: " + str(accuracy_train))
    return model, accuracy_train, end_train_time - start_train_time


def test_model(model, X_test, y_test):
    print("test model")
    start_test_time = time.time()
    try:
        if isinstance(model, sklearn.neural_network.MLPClassifier):
            # X_test = X_test.reshape(X_test.shape[0], -1)
            print("set y_pred_test_classes")
            y_pred_test_classes = model.predict(X_test)
            print("y_pred_test_classes init")
        else:
            y_pred_test = model.predict(X_test)
            print("set y_pred_test_classes")
            y_pred_test_classes = (y_pred_test > 0.5).astype(int)
            print("y_pred_test_classes init")
    except:
        y_pred_test = model.predict(X_test)
        print("set y_pred_test_classes")
        y_pred_test_classes = (y_pred_test > 0.5).astype(int)
        print("y_pred_test_classes init")
    end_test_time = time.time()
    print("Predicted test classes:", np.unique(y_pred_test_classes, return_counts=True))

    # Calculate metrics for test
    cm = confusion_matrix(y_test, y_pred_test_classes)
    print("Confusion matrix:\n", cm)
    if cm.shape == (1, 1):
        if y_pred_test[0] == 0:  # or check y_train[0] if that's more appropriate
            tn = cm[0][0]
            fp = fn = tp = 0
        else:
            tp = cm[0][0]
            tn = fp = fn = 0
    else:
        tn, fp, fn, tp = cm.ravel()

    if (tn + fp) == 0:
        specificity_test = 1
    else:
        specificity_test = tn / (tn + fp)
    sensitivity_test = recall_score(y_test.ravel(), y_pred_test_classes.ravel())
    f1_test = f1_score(y_test.ravel(), y_pred_test_classes.ravel())
    accuracy_test = accuracy_score(y_test.ravel(), y_pred_test_classes.ravel())
    return y_pred_test_classes, accuracy_test, specificity_test, sensitivity_test, f1_test, end_test_time - start_test_time, tp, tn, fp, fn


def build_model(algo, window_size=None, num_of_features=None, encoding_length=None):
    model = None
    if algo == 'dnn':
        model = Sequential()
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    elif algo == 'cnn':
        kernel_size = 3
        if num_of_features < 3 or window_size < 3:
            kernel_size = min(window_size, num_of_features)
        model = Sequential()
        model.add(Conv2D(32, input_shape=(window_size, num_of_features, encoding_length), kernel_size=kernel_size,
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=1))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    elif algo == 'lstm':
        model = Sequential()
        model.add(LSTM(32, input_shape=(window_size, num_of_features)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    elif algo == 'mlp':
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(hidden_layer_sizes=(64, 32),
                              solver='adam',
                              activation='relu',
                              max_iter=Epochs)
    return model


def ret_vec(value):
    # Debug print
    # print(f"Value before processing: {value}, Type: {type(value)}")

    if isinstance(value, bytes):
        try:
            value = value.decode('utf-8')  # Decoding bytes to string
        except Exception as e:
            print(f"Error decoding value: {e}")

    zero_array = np.zeros(300, dtype=np.float32)
    # Another debug print
    # print(f"Value after potential decoding: {value}, Type: {type(value)}")
    if not isinstance(value, str):
        return np.zeroszero_array
    value = ' '.join(camel_case_split(value))
    vec = re.split(',| |_|-', value)

    vec = [x for x in vec if x]
    if vec:
        # print(vec)
        vec300 = np.array(list(map(lambda x: ft.get_word_vector(x), vec)))
        # print(vec300)
        meanVec = np.mean(vec300, axis=0)
        return meanVec
    return zero_array


from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

import math


def is_numeric(value):
    try:
        # Convert value to float and check if it is NaN
        numeric_value = float(value)
        if math.isnan(numeric_value):
            return False
        return True
    except ValueError:
        return False


def is_nan(value):
    try:
        # Convert value to float and check if it is NaN
        numeric_value = float(value)
        if math.isnan(numeric_value):
            return True
        return False
    except ValueError:
        return False


def BertEncode(df, numeric_c, encoding_cache=None):
    print("start bert encoding")

    if encoding_cache is None:
        encoding_cache = {}

    # This will store our modified columns
    modified_df = pd.DataFrame()

    for col_name in df.columns:
        if col_name in ['malicious', 'Process Name', 'Ransomware'] or col_name in numeric_c:
            print(f"Skipping column {col_name} as it's in the exclusion list.")
            continue

        print(f"start encode: {col_name} col")

        # Convert all values to string and replace NaN with empty strings
        df[col_name] = df[col_name].astype(str).fillna('')

        col_data = []

        for value in df[col_name]:
            if is_nan(value):
                col_data.append([0] * 768)
                continue
            if is_numeric(value):
                # If the value is numeric, add it at the end of a list of zeros
                numeric_value = float(value)
                col_data.append([0] * 767 + [numeric_value])
                continue
            sepValue = camel_case_split(value)
            value = ' '.join(sepValue).lower()
            if value in encoding_cache:
                col_data.append(encoding_cache[value])
                continue

            inputs = tokenizer(value, padding=True, truncation=True, return_tensors="pt", max_length=512)
            outputs = model(**inputs)

            cls_output = outputs.last_hidden_state[:, 0, :].detach().numpy().squeeze().astype(np.float32)

            encoding_cache[value] = cls_output
            col_data.append(cls_output)

        # Convert list of numpy arrays into a series, then assign to our new dataframe
        modified_df[col_name] = pd.Series(col_data)

    # Reset the index of both DataFrames
    df = df.reset_index(drop=True)
    modified_df = modified_df.reset_index(drop=True)

    # Concatenate modified columns with numeric columns and columns we skipped
    for col_name in df.columns:
        if col_name not in modified_df.columns:
            modified_df[col_name] = df[col_name]

    return modified_df, encoding_cache


def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]


def check_dtypes_and_handle_nans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check and handle data types and NaN values in a dataframe.

    Parameters:
    - df (pd.DataFrame): Input dataframe

    Returns:
    - pd.DataFrame: Cleaned dataframe
    """

    # Check data types
    dtypes = df.dtypes
    print("Data types before handling:")
    print(dtypes)

    # Convert columns with object dtype to string
    object_cols = dtypes[dtypes == 'object'].index.tolist()
    for col in object_cols:
        df[col] = df[col].astype(str)

    # Handle NaN values (you can adapt this depending on your needs)
    # Here, I'm replacing NaN values with a placeholder string for object columns and with 0 for numeric columns
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna('', inplace=True)  # or other placeholder string
        else:
            df[col].fillna(0, inplace=True)  # replace NaN with 0

    # Confirm NaN handling
    nan_counts = df.isna().sum()
    print("\nNumber of NaN values after handling:")
    print(nan_counts)

    return df


# Usage:
# df = check_dtypes_and_handle_nans(df)


def W2v(df, numeric_c):
    t_empty_list = np.zeros(300, dtype=np.float32)

    for col_name in df.columns:
        if col_name in ['malicious', 'Process Name', 'Ransomware'] or col_name in numeric_c:
            continue

        # Convert all values to string and replace NaN with empty strings
        df[col_name] = df[col_name].astype(str).fillna('')
        # print(col_name)
        col_data = []

        for value in df[col_name]:
            # Append the numpy array to the list
            # print(value)
            if is_numeric(value):
                # If the value is numeric, add it at the end of a list of zeros
                numeric_value = float(value)
                col_data.append([0] * 299 + [numeric_value])
                continue
            if value in ["nan", "0.0", "NaN"]:
                col_data.append(t_empty_list)
            else:
                vec = ret_vec(value)
                col_data.append(vec)
                # print(f"Shape of vec: {np.array(vec).shape}")  # Debug print
        print(f"Shape of col_data for {col_name}: {np.array(col_data).shape}")  # Debug print
        # Assign the list of numpy arrays to the DataFrame column
        df[col_name] = np.array(col_data)
        print(df[col_name].shape)
    print(df.shape)

    return df


def run_experiments_for_different_models(window_size, syscallsAmount, train_data, y_train, test_data, y_test, features,
                                         skip, encoding, op):
    # Create windows
    fixed_window_count = (syscallsAmount - window_size) // skip + 1

    print("start make windows")
    X_train_windows, y_train_windows = make_windows_per_process(train_data, y_train,
                                                                train_data['Process Name'], window_size,
                                                                skip, fixed_window_count, encoding)
    X_test_windows, y_test_windows = make_windows_per_process(test_data, y_test, test_data['Process Name'],
                                                              window_size, skip, fixed_window_count, encoding)
    print("end make windows")

    print("Train windows class distribution:", np.unique(y_train_windows, return_counts=True))
    print("Test windows class distribution:", np.unique(y_test_windows, return_counts=True))

    print("X_train_windows shape:", X_train_windows.shape)
    print("y_train_windows shape:", y_train_windows.shape)
    print("X_test_windows shape:", X_test_windows.shape)
    print("y_test_windows shape:", y_test_windows.shape)

    for algo in ['cnn', 'dnn', 'lstm', 'mlp']:  # 'cnn', 'dnn', 'lstm', 'mlp'

        # print("sleep for cool down")
        # time.sleep(180)
        # print("wake up")
        try:
            print(f"---- model type : {algo}")
            if encoding == 'OH':
                X_train_windows = X_train_windows[:, :, np.newaxis, :]
                X_test_windows = X_test_windows[:, :, np.newaxis, :]
            # Reshaping data based on model type
            if algo == 'lstm':
                X_train_windows = X_train_windows.reshape(X_train_windows.shape[0], X_train_windows.shape[1], -1)
                X_test_windows = X_test_windows.reshape(X_test_windows.shape[0], X_test_windows.shape[1], -1)
                print("--SHAPE CHANGED--")
                print("X_train_windows shape:", X_train_windows.shape)
                print("y_train_windows shape:", y_train_windows.shape)

            # sep if
            if algo == 'mlp':
                X_train_windows = X_train_windows.reshape(X_train_windows.shape[0], -1)
                X_test_windows = X_test_windows.reshape(X_test_windows.shape[0], -1)
                print("--SHAPE CHANGED--")
                print("X_train_windows shape:", X_train_windows.shape)
                print("y_train_windows shape:", y_train_windows.shape)
                print("X_test_windows shape:", X_test_windows.shape)
                print("y_test_windows shape:", y_test_windows.shape)
                model = build_model(algo)
            # Building model based on model type
            if algo == 'mlp':
                model = build_model(algo, X_train_windows.shape[1])
            elif algo == 'lstm':
                model = build_model(algo, X_train_windows.shape[1], X_train_windows.shape[2])
            else:
                model = build_model(algo, X_train_windows.shape[1], X_train_windows.shape[2],
                                    X_train_windows.shape[3])

            model, accuracy_train, train_time = train_model(model, X_train_windows, y_train_windows)
            K.clear_session()
            gc.collect()
            predictions, accuracy_test, specificity_test, sensitivity_test, f1_test, test_time, tp, tn, fp, fn = test_model(
                model,
                X_test_windows,
                y_test_windows)

            print("Training accuracy:", accuracy_train)
            print(f"Test accuracy: {accuracy_test}")
            print(f"Train time: {train_time}")

            # Save the trained model
            save_path = create_saving_path(algo, window_size, features, encoding, op)
            save_predictions_to_csv(y_test_windows, predictions, accuracy_test, syscallsAmount, save_path)
            save_model(model, algo, syscallsAmount, skip, save_path)

            save_results(algo, accuracy_train, accuracy_test, sensitivity_test, specificity_test, f1_test, train_time,
                         test_time,
                         X_train_windows, X_test_windows, tp, tn, fp, fn, syscallsAmount, features, skip, encoding,
                         window_size, op)

        except Exception as exception:
            print(exception)
            print(f"SKIP TEST: {algo}")
            print(f"amoutOfSystemcalls : {syscallsAmount}")
            save_results(algo, "failed", "failed", "failed", "failed", "failed", "failed",
                         "failed",
                         X_train_windows, X_test_windows, "failed", "failed", "failed", "failed", syscallsAmount,
                         features, skip, encoding, window_size, op)
            K.clear_session()
            if 'model' in locals():
                del model
            gc.collect()
            raise Exception(f"failed at window_size: {window_size}")
        K.clear_session()
        del model
        gc.collect()


def run_experiments_for_specific_window_size_different_skips(window_size, syscallsAmount, train_data, y_train,
                                                             test_data, y_test, features, encoding, op):
    # if window_size > 6:
    #     SKIP = 6
    # else:
    #     SKIP = window_size
    SKIP = window_size
    # SKIP = 1
    # for skip in range(window_size, window_size+1):
    try:
        run_experiments_for_different_models(window_size, syscallsAmount, train_data, y_train, test_data, y_test,
                                             features, SKIP, encoding, op)
    except Exception as exception:
        print(exception)
        print(exception.__traceback__)
        # break


def save_results(algo, accuracy_train, accuracy_test, sensitivity_test, specificity_test, f1_test, train_time,
                 test_time,
                 X_train_windows, X_test_windows, tp, tn, fp, fn, syscallsAmount, features, SKIP, encoding, window_size,
                 op):
    print("save results in DF")
    results['amoutOfSystemcalls'].append(syscallsAmount)
    results['Features'].append('&'.join(features))
    results['Model'].append(algo)
    results['Train_Accuracy'].append(accuracy_train)
    results['Test_Accuracy'].append(accuracy_test)
    results['Sensitivity'].append(sensitivity_test)
    results['Specificity'].append(specificity_test)
    results['F1_Score'].append(f1_test)
    results['Train_Time'].append(train_time)
    results['Test_Time'].append(test_time)
    results['Train_Set_Shape'].append(X_train_windows.shape)
    results['Test_Set_Shape'].append(X_test_windows.shape)
    results['TP'].append(tp)
    results['TN'].append(tn)
    results['FP'].append(fp)
    results['FN'].append(fn)
    results['encoding'].append(encoding)
    results['skip'].append(SKIP)
    results['win_size'].append(window_size)
    results['op'].append(op)
    print(results)


import joblib


def create_saving_path(algo, window_size, features, encoding, op):
    """
    Save the trained model in the appropriate directory.

    :param model: Trained model to be saved.
    :param algo: Algorithm type ('cnn', 'dnn', 'lstm', 'mlp').
    :param window_size: Size of the window.
    :param features: Features the model trained on.
    :param syscallsAmount: The length of the syscall it trained on.
    """
    # Base path where models are saved
    base_path = f'E:\\MSc\\result_saved_models_new\\{op}'
    features_str = "_".join(features).lower()

    directory_path = os.path.join(base_path, encoding.upper(), algo.upper(), str(window_size), features_str)

    return directory_path


def save_model(model, algo, syscallsAmount, skip, directory_path):
    """
    Save the trained model in the appropriate directory.

    :param model: Trained model to be saved.
    :param algo: Algorithm type ('cnn', 'dnn', 'lstm', 'mlp').
    :param window_size: Size of the window.
    :param features: Features the model trained on.
    :param syscallsAmount: The length of the syscall it trained on.
    """
    if algo.lower() == 'mlp':
        model_name = f"{syscallsAmount}_skip{skip}_50epoch.pkl"
    else:
        model_name = f"{syscallsAmount}_skip{skip}_50epoch.h5"

    # Full path to save the model
    save_path = os.path.join(directory_path, model_name)
    # Save the model
    if algo.lower() == 'mlp':
        joblib.dump(model, save_path)
    else:
        model.save(save_path)
    print(f"Model saved to {save_path}")


def save_predictions_to_csv(y_test, predictions, accuracy_test, syscallsAmount, directory_path):
    """
    Save the predictions and accuracy test to files in the specified directory structure.
    """
    try:
        # Flatten y_test and predictions to ensure they are 1-dimensional
        y_test_flat = np.ravel(y_test)
        predictions_flat = np.ravel(predictions)

        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        csv_file_path = os.path.join(directory_path, f"predictions_{str(syscallsAmount)}50epoch.csv")
        txt_file_path = os.path.join(directory_path, f"accuracy_test_{str(syscallsAmount)}_50epoch.txt")

        # Convert to DataFrame and save
        df = pd.DataFrame({'Actual': y_test_flat, 'Predictions': predictions_flat})
        df.to_csv(csv_file_path, index=False)
        print(f"Predictions saved to {csv_file_path}")

        # Save accuracy_test to TXT
        with open(txt_file_path, 'w') as file:
            file.write(f"Accuracy Test: {accuracy_test}")
        print(f"Test accuracy saved to {txt_file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
        exit()


def run_experiments_for_features_and_amount_of_systemcalls(features, syscallsAmount, encoding, op):
    global bert_encoding_cache
    # Load data
    print("Load data")
    train_data, test_data = load_data("E:\\MSc\\viruses\\100v\\reselts2\\reselts2\\extracted",
                                      "E:\\MSc\\benign\\final\\final",
                                      "train49.txt",
                                      "test13.txt",
                                      syscallsAmount,
                                      features,
                                      op)

    if 'Detail' in features:
        train_data = separate_detail_column(train_data)
        column_order = train_data.columns.tolist()
        test_data = separate_detail_column(test_data)
        test_data = test_data[column_order]

    print("train_data columns")
    print(train_data.columns)
    print("test_data columns")
    print(test_data.columns)

    train_data, numeric_cols_trian = norm_data(train_data, encoding)
    print("numeric cols train: " + str(numeric_cols_trian))
    test_data, numeric_cols_test = norm_data(test_data, encoding)
    print("numeric cols test: " + str(numeric_cols_test))

    if encoding == 'FT':
        print("W2v train_data")
        train_data = W2v(train_data, numeric_cols_trian)
        print("W2v test_data")
        test_data = W2v(test_data, numeric_cols_test)
    elif encoding == 'BERT':
        print("bert train_data")
        train_data, bert_encoding_cache = BertEncode(train_data, numeric_cols_trian, bert_encoding_cache)
        print("bert test_data")
        test_data, bert_encoding_cache = BertEncode(test_data, numeric_cols_trian, bert_encoding_cache)
    elif encoding == 'OH':
        if 'Duration' in features or 'Detail' in features:
            return
        train_data, test_data = oneHotEncode(train_data, test_data, features)
    else:
        raise Exception('unknown encoding')

    # Define the target
    y_train = train_data['Ransomware'].values
    train_data = train_data.drop(columns=['Ransomware'])
    y_test = test_data['Ransomware'].values
    test_data = test_data.drop(columns=['Ransomware'])

    print("Train class distribution:", np.unique(y_train, return_counts=True))
    print("Test class distribution:", np.unique(y_test, return_counts=True))
    print("train_data shape:", train_data.shape)
    print("y_train shape:", y_train.shape)
    print("test_data shape:", test_data.shape)
    print("y_test shape:", y_test.shape)
    # Parameters for the experiment
    window_sizes = [1, 3, 5, 7]  # 1, 3, 5, 7, 9

    for window_size in window_sizes:
        print(f"--- window_size : {window_size}")
        run_experiments_for_specific_window_size_different_skips(window_size, syscallsAmount, train_data, y_train,
                                                                 test_data, y_test, features, encoding, op)


def oneHotEncode(train_data, test_data, features):
    def encode_and_concat(df, encoder, column_name):
        # Transforming the data using the fitted encoder
        transformed = encoder.transform(df[[column_name]])
        # Creating a DataFrame with the transformed data
        transformed_df = pd.DataFrame(transformed, columns=encoder.get_feature_names_out([column_name]),
                                      index=df.index)
        # Concatenating the original dataframe (minus the encoded column) with the new encoded dataframe
        return pd.concat([df.drop(columns=[column_name]), transformed_df], axis=1)

    le_operation_result = OneHotEncoder(sparse=False)  # Use sparse=False to get a dense matrix

    if 'Result' in features:
        train_data['Operation_Result'] = train_data['Operation'].astype(str) + "_" + train_data['Result'].astype(str)
        train_data = train_data.drop(columns=['Operation', 'Result'])
        test_data['Operation_Result'] = test_data['Operation'].astype(str) + "_" + test_data['Result'].astype(str)
        test_data = test_data.drop(columns=['Operation', 'Result'])
        all_data = pd.concat([train_data, test_data])
        le_operation_result.fit(all_data[['Operation_Result']])
        train_data = encode_and_concat(train_data, le_operation_result, 'Operation_Result')
        test_data = encode_and_concat(test_data, le_operation_result, 'Operation_Result')

    elif len(features) == 1 and 'Operation' in features:
        all_data = pd.concat([train_data, test_data])
        # Initializing OneHotEncoder
        le_operation_result.fit(all_data[['Operation']])

        train_data = encode_and_concat(train_data, le_operation_result, 'Operation')
        test_data = encode_and_concat(test_data, le_operation_result, 'Operation')
    else:
        print("unknown feature")
        print(features)
        exit()

    return train_data, test_data


# Function to run experiments
def run_experiments():
    features_list = [['Operation'], ['Operation', 'Result'], ['Operation', 'Result', 'Duration'],
                     ['Operation', 'Result', 'Duration',
                      'Detail']]  # ['Operation'], ['Operation', 'Result'], ['Operation', 'Result', 'Duration'], ['Operation', 'Result', 'Duration', 'Detail']
    amoutOfSystemcalls = [500, 1000, 2000, 3000, 4000, 5000]  # 500, 1000, 2000, 3000, 4000, 5000
    test_set_split_options = [0]
    for test_set_split in test_set_split_options:
        for features in features_list:
            print(f"- features: {features}")
            for syscallsAmount in amoutOfSystemcalls:
                print(f"-- amoutOfSystemcalls : {syscallsAmount}")

                if 'Duration' not in features and 'Detail' not in features:
                    print("------------")
                    print("OH Encoding")
                    print("------------")
                    run_experiments_for_features_and_amount_of_systemcalls(features, syscallsAmount, 'OH',
                                                                           test_set_split)

                if syscallsAmount > 500 and 'Detail' in features:
                    continue

                if syscallsAmount > 3000 and 'Duration' in features:
                    continue
                print("------------")
                print("BERT Encoding")
                print("------------")
                run_experiments_for_features_and_amount_of_systemcalls(features, syscallsAmount, 'BERT', test_set_split)

                print("------------")
                print("FT Encoding")
                print("------------")
                run_experiments_for_features_and_amount_of_systemcalls(features, syscallsAmount, 'FT')
        print(f"save results in csv op: {test_set_split}")
        results_df = pd.DataFrame(results)
        results_df.to_csv(
            f'experiments_results_new/experiment_results_different_test_set_op_{test_set_split}_30Epochs_detils2.csv',
            index=False)


try:
    run_experiments()
except Exception as e:
    print(e)
finally:
    print("save results in csv")
    # results_df = pd.DataFrame(results)
    # results_df.to_csv(f'2experiments_results_new/experiment_results_different_test_set_moreEpoch.csv', index=False)
