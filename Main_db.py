import base64
from flask import Flask, request, render_template, session, redirect, url_for, g, jsonify, flash
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import os
import pickle
from xgboost import XGBRegressor, XGBClassifier
from tpot import TPOTClassifier, TPOTRegressor
from threading import Thread
import json
import io
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from collections import OrderedDict
import operator
from datetime import datetime
import mysql.connector
from contextlib import contextmanager
import time
import sys
from io import StringIO
from werkzeug.utils import secure_filename


def contents(FILE_NAME):
    x_in = pd.read_csv(FILE_NAME)
    non_selectable_fields = {}
    selectable_fields = {}
    for field in x_in.columns:
        if x_in[field].dtype == "object":
            if x_in[field].unique().size >= 10:
            # if x_in[field].unique().size >= 0.5 * x_in[field].notnull().sum():
                non_selectable_fields[field] = x_in[field].dtype
            else:
                selectable_fields[field] = x_in[field].dtype
        else:
            selectable_fields[field] = x_in[field].dtype

    return non_selectable_fields, selectable_fields


def pre_check(data, output_var):
    output_col = data[output_var]
    # Restrict to 20 columns
    data = data.iloc[:, 0:20]
    data[output_var] = output_col
    # Drop any record with a NA value
    data = data.dropna()
    # Restrict to 1000 rows
    if len(data) > 1000:
        data = data.head(1000)
    return data


def field_info(data):
    Field_info = {}
    for field in data.columns:
        if data[field].dtype != "object" and data[field].dtype != "bool":
            # store the range values of the numeric field type
            Field_info[field] = {'min': data[field].describe()['min'], 'max': data[field].describe()['max']}
            # print(Model_info)
        if data[field].dtype == 'bool':
            Field_info[field] = {'min': '0', 'max': '1'}
    # print(Field_info)
    return Field_info


def data_preprocessing(data, fields,output_var):
    data = pre_check(data, output_var)
    for field in data.columns:
        # check if the column is in the selected fields list
        if field in fields:
            # check if the column is of categorical type
            if data[field].dtype == "object":
                print("field",field)
                print("output_var",output_var[0])
                if field == output_var[0]:
                    print("inside field compare")
                    print("need dummy", field)
                    dummydf = pd.get_dummies(data.loc[:, [field]])
                    dummydf_single_column_name = dummydf.columns[0]
                    dummydf = dummydf.iloc[:, 0]
                    data.drop([field], inplace=True, axis=1)
                    data = pd.concat([data, dummydf], axis=1)

                    data.rename(columns={dummydf_single_column_name: output_var[0]}, inplace=True)
                    print("concatenated columns", data.columns)

                else:
                    print("need dummy", field)
                    dummydf = pd.get_dummies(data.loc[:, [field]])
                    print("dummdf",(dummydf))
                    data.drop([field], inplace=True, axis=1)
                    print("dummified", data)
                    data = pd.concat([data, dummydf], axis=1)
                    print("concatenated columns", data.columns)
        else:
            data.drop([field], inplace=True, axis=1)
    return data


class CustomThread(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return


def get_feature_importances(task):
    print("inside feature function and img")
    img = io.BytesIO()
    print("after img function")

    x = pd.read_csv('x.csv')
    y = pd.read_csv('y.csv')
    print("after csv read function")

    if task == 'Classification':
        model = XGBClassifier(n_jobs=-1)
    else:
        model = XGBRegressor(n_jobs=-1)
    print("before fit csv")

    model.fit(x, y)
    print("after fit")
    importances = model.feature_importances_
    print(type(importances))
    print(importances)

    plt.figure(figsize=(10, 10))
    plt.title('Most Important Features')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    print("after plt")

    feature_names = list(x.columns)
    feat_importances = dict(zip(feature_names, importances))
    feat_importances = OrderedDict(sorted(feat_importances.items(), key=operator.itemgetter(1), reverse=True)[:5])
    print("ordered dict")

    plt.xticks(range(len(feat_importances)), list(feat_importances.keys()))
    plt.bar(range(len(feat_importances)), list(feat_importances.values()), align='center')
    plt.savefig(img, format='png')
    print("after oplt save")

    img.seek(0)
    print("after img seek ")

    graph_url = base64.b64encode(img.getvalue()).decode()
    print("after graph url")

    plt.close()
    return 'data:image/png;base64,{}'.format(graph_url), feat_importances


def evaluate_model(test_x, test_y, task, ID):
    print("inside evaluate model")

    metrics = {}
    # Prediction using tpot pickled model

    with open('Files/model_' + str(ID) + '/model_' + str(ID) + '.pkl', 'rb') as mod:
        loaded_model = pickle.load(mod)

    # update database
    with db_connection() as db:

        cursor = db.cursor()
        cursor.execute("select database();")
        database = cursor.fetchone()
        cursor.execute(
            "Update automl.status Set Status = %s Where ID = %s",
            ("Evaluated", ID))
        cursor.execute("SELECT * FROM automl.status")
        table = cursor.fetchall()

        db.commit()
        cursor.close()

    predy = loaded_model.predict(test_x)
    print(predy)

    if task == "Classification":

        metrics['task'] = 'Classification'
        metrics['prediction'] = predy.tolist()
        metrics['accuracy'] = int(100 * accuracy_score(predy, test_y))
        metrics['precision'] = int(100 * precision_score(predy, test_y, average="weighted"))
        metrics['recall'] = int(100 * recall_score(predy, test_y, average="weighted"))
        metrics['f1'] = int(100 * f1_score(predy, test_y, average="weighted"))

    else:

        metrics['task'] = 'Regression'
        metrics['r2_score'] = r2_score(predy, test_y)
        metrics['mean_absolute_error'] = mean_absolute_error(predy, test_y)
        metrics['mean_squared_error'] = mean_squared_error(predy, test_y)

    metrics['model_name'] = type(loaded_model[-1]).__name__
    print("before the feature importance")
    feature_plot, feat_importances = get_feature_importances(task)

    print("feature important", str(feat_importances))
    print("type of feature important", type(str(feat_importances)))

    metrics['feat_importances'] = str(feat_importances)
    metrics['feature_plot'] = feature_plot

    return metrics


def create_db_connection():
    return mysql.connector.connect(
        host="localhost", user="root",
        password="", database="automl")


@contextmanager
def db_connection():
    db = create_db_connection()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def capture_tpot_output():
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    try:
        yield captured_output
    finally:
        sys.stdout = old_stdout


def AutoML(input_vars, output_var, data, task, ID, final_columns, db):
    start_time = time.time()
    print("inside automl")
    x = data.drop(output_var, axis=1)
    y = data[output_var]

    # split into train and test sets
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=.7, random_state=120)

    x.to_csv('x.csv', index=False)
    y.to_csv('y.csv', index=False)
    # test_y.to_csv('test_y.csv')
    with db_connection() as db:
        cursor = db.cursor()
        cursor.execute("select database();")
        record = cursor.fetchone()
        try:

            if task == "Classification":
                tpot_model = TPOTClassifier(generations=2, verbosity=3, max_time_mins=2, max_eval_time_mins=0.04,
                                            population_size=40)

            else:
                tpot_model = TPOTRegressor(generations=1, verbosity=2, max_time_mins=2, max_eval_time_mins=0.04,
                                           population_size=40)
            print("before the running status")
            # update db

            cursor.execute(
                "Update automl.status Set Status = %s Where ID = %s",
                ("Running", ID))
            cursor.execute("SELECT * FROM automl.status")
            table = cursor.fetchall()
            db.commit()

            tpot_model.fit(train_x, train_y)

            end_time = time.time()
            train_time = end_time - start_time

            tpot_score = tpot_model.score(train_x, train_y)
            print(tpot_score)
            tpot_model.export('tpot_digits_pipeline.py')
            message = "The top performing pipeline is saved to the file 'tpot_digits_pipeline.py'"
            fitted_pipeline = tpot_model.fitted_pipeline_
            print(fitted_pipeline)

        except:
            # update database
            print("inside the except block")
            cursor.execute("Update automl.status Set Status = %s Where ID = %s", ("Failure", ID))
            cursor.execute("SELECT * FROM automl.status")
            table = cursor.fetchall()
            db.commit()

        Model_info = {"Input Variables": x.columns.tolist(),  # input_vars,
                      "Tpot Score": tpot_score,
                      "Best Pipeline": message}

        print("before evaluate model")

        with open('Files/model_' + str(ID) + '/model_info_' + str(ID) + '.json', "w") as file:
            json.dump(Model_info, file)

        with open('Files/model_' + str(ID) + '/model_' + str(ID) + '.pkl', 'wb') as mod:
            pickle.dump(fitted_pipeline, mod)

        # evaluate your model
        metrics = evaluate_model(test_x, test_y, task, ID)

        x_inputs_str = ','.join(final_columns)
        print("after evaluated")
        # print(metrics)
        if len(metrics) != 0:
            print("inside success updation")
            if metrics['task'] == 'Classification':

                # update database
                cursor.execute(
                    "Update automl.status Set Status = %s, Model_name = %s, Model_accuracy= %s, Model_recall= %s, Model_precision= %s, Model_f1= %s, Input_variables= %s, Important_variables=%s, feature_plot=%s, Train_time=%s Where ID = %s",
                    ("Success", metrics['model_name'], metrics['accuracy'], metrics['recall'], metrics['precision'],
                     metrics['f1'],
                     x_inputs_str, metrics['feat_importances'], metrics['feature_plot'], train_time, ID))
            elif metrics['task'] == 'Regression':
                # update database
                cursor.execute(
                    "Update automl.status Set Status = %s, Model_name = %s, Model_r2_score= %s, Model_mean_absolute_error= %s, Model_mean_squared_error= %s, Input_variables= %s, Important_variables=%s, feature_plot=%s, Train_time=%s Where ID = %s",
                    ("Success", metrics['model_name'], metrics['r2_score'], metrics['mean_absolute_error'],
                     metrics['mean_squared_error'],
                     x_inputs_str, metrics['feat_importances'], metrics['feature_plot'], train_time, ID))

            cursor.execute("SELECT * FROM automl.status")
            table = cursor.fetchall()
            db.commit()
            cursor.close()

        with open('Files/model_' + str(ID) + '/metrics_info_' + str(ID) + '.json', "w") as file:
            json.dump(metrics, file)


app = Flask(__name__)
app.secret_key = "any random string"
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000
ALLOWED_EXTENSIONS = {'csv'}


def get_db():
    db = getattr(g, "_database", None)
    if db is None or not db.is_connected():
        db = g._database = mysql.connector.connect(
            host="localhost", user="root",
            password="", database="automl")
    return db


@app.teardown_appcontext
def teardown_db(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()


# cache = SimpleCache()

@app.route('/', methods=['GET'])
def index():
    db = get_db()
    cursor = db.cursor()

    # cursor = con.cursor()
    cursor.execute("select database();")
    record = cursor.fetchone()
    print(record)
    cursor.execute("SELECT MAX(ID) FROM automl.status")
    ID = cursor.fetchone()[0]
    cursor.close()
    session['Hist_ID'] = ID
    print("history_id", ID)
    return redirect(url_for('home'))


@app.route('/home', methods=['GET'])
def home():
    db = get_db()
    cursor = db.cursor()
    cursor.execute("select database();")
    record = cursor.fetchone()
    print(record)
    Hist_ID = session.get('Hist_ID')
    if Hist_ID is None:
        Hist_ID = -1

    print(Hist_ID)
    options = []
    reader = pd.read_csv('Resources/Industries_list.csv')
    print(reader.columns)
    for index, row in reader.iterrows():
        options.append(row[0])


    print("options", options)

    cursor.execute("SELECT * FROM automl.status")
    result = cursor.fetchall()
    cursor.execute("SELECT COUNT(*) FROM automl.status")
    num_of_models = cursor.fetchone()[0]
    cursor.execute("SELECT AVG(Train_time) FROM automl.status")

    print("outside")
    avg = cursor.fetchone()
    if len(avg) != 0:
        print(avg)
        avg_train_time = round(avg[0], 2)
        print(avg_train_time)
    else:
        avg_train_time = 'Nil'
    print(avg_train_time)

    cursor.close()
    return render_template('file_upload_3.html', tasks=["Classification", "Regression"], result=result, Hist_ID=Hist_ID,
                           num_of_models=num_of_models, avg_train_time=avg_train_time, options=options)


@app.route('/get_chart_data')
def get_chart_data():
    db = get_db()
    cursor = db.cursor()
    cursor.execute("select database();")
    record = cursor.fetchone()
    print(record)

    cursor.execute("SELECT Domain,Category,COUNT(*) FROM automl.status Group BY Domain,Category")
    Dom_rec = cursor.fetchall()
    print("domain frequency")
    print(Dom_rec)
    data_dict = {
        'domain': [],
        'Classification': [],
        'Regression': []
    }

    for domain, task, count in Dom_rec:
        if domain not in data_dict['domain']:
            data_dict['domain'].append(domain)
            data_dict['Classification'].append(0)
            data_dict['Regression'].append(0)
        domain_index = data_dict['domain'].index(domain)
        if task == 'Classification':
            data_dict['Classification'][domain_index] = count
        elif task == 'Regression':
            data_dict['Regression'][domain_index] = count

    # Convert the lists to the desired format
    data = {
        'domain': data_dict['domain'],
        'Classification': data_dict['Classification'],
        'Regression': data_dict['Regression']
    }
    print(data)

    x_data = np.array(Dom_rec)[:, 0]
    y_data = np.array(Dom_rec)[:, 1]
    print(list(x_data))
    print(list(y_data))
    cursor.close()

    return jsonify(data)


@app.route('/get_data')
def get_data():
    db = get_db()
    cursor = db.cursor()

    cursor.execute("select database();")
    record = cursor.fetchone()
    print(record)
    # delete the Sales table contents if there exists
    # cursor.execute("Delete from automl.status")
    cur_ID = session.get('Current_ID')
    Hist_ID = session.get('Hist_ID')
    cursor.execute("SELECT * FROM automl.status WHERE ID > %s", (Hist_ID,))
    print("current ID inside getdata", cur_ID)
    curr_result = cursor.fetchall()

    row_headers = [x[0] for x in cursor.description]  # this will extract row headers

    json_data = []
    for r in curr_result:
        json_data.append(dict(zip(row_headers, r)))
    cursor.close()
    print("inside the getdata function")
    return jsonify(json_data)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            status = "Submitted"
            flash('File was successfully uploaded!', 'success')

            task = request.form.get('task')
            print(type(task))
            domain = request.form.get('domain')
            print(domain.title())
            print(type(domain))
            session["task"] = task

            current_date = datetime.now()
            formatted_date = current_date.strftime("%Y/%m/%d %H:%M:%S")

            db = get_db()
            cursor = db.cursor()
            cursor.execute("select database();")
            record = cursor.fetchone()
            print(record)
            # delete the Sales table contents if there exists
            # cursor.execute("Delete from automl.status")
            cursor.execute("INSERT INTO automl.status(Date, Category, Domain, Status) VALUES(%s, %s,%s,%s)",
                           (formatted_date, task, domain.title(), status))
            cursor.execute("SELECT MAX(ID) FROM automl.status")
            ID = cursor.fetchone()
            print(ID)
            print(ID[0])
            print(type(ID[0]))
            session['Current_ID'] = ID[0]

            filename = secure_filename(file.filename)
            new_filename = f'{filename.split(".")[0]}_{"dataset"}.csv'
            if not os.path.exists("Files/model_" + str(ID[0])):
                # if the demo_folder directory is not present then create it.
                os.makedirs("Files/model_" + str(ID[0]))
            save_location = os.path.join('Files/model_' + str(ID[0]), new_filename)
            file.save(save_location)
            session["train_file"] = save_location
            non_selectable_fields, selectable_fields = contents(save_location)
            print(non_selectable_fields, selectable_fields)
            cursor.execute("SELECT * FROM automl.status")
            table = cursor.fetchall()
            db.commit()
            cursor.close()

            return render_template('select_variables_3.html', non_selectable_fields=non_selectable_fields,
                                   selectable_fields=selectable_fields)
        else:
            print("incorrect format")
            flash('Invalid file format needs to be .csv', 'danger')
            return redirect(url_for('home'))
    else:
        print("not uploaded")
        flash('File was not successfully uploaded!', 'danger')
        return redirect(url_for('home'))


@app.route('/select_variables', methods=['POST'])
def select_variables():
    if request.method == "POST":
        selected_output = request.form.getlist('output_variable')
        selected_inputs = request.form.getlist('input_variables')
        task = session.get('task')

        print(selected_output, selected_inputs)

        # Start the AUTOML process here
        file = session.get('train_file')
        x_in = pd.read_csv(file)

        finalfile = data_preprocessing(x_in, selected_inputs + selected_output, selected_output)
        print("finallllll col", finalfile.columns)
        # print(type(finalfile.columns))
        x_inputs = finalfile.drop(selected_output, axis=1)

        ID = session.get('Current_ID')

        Field_info = field_info(x_inputs)
        print("FIELD_INFO", Field_info)
        with open('Files/model_' + str(ID) + '/field_info_' + str(ID) + '.json', "w") as file:
            json.dump(Field_info, file)

        print("final file info", finalfile.info())
        print("final file describe", finalfile.describe())
        db = get_db()

        global t1
        t1 = CustomThread(target=AutoML, args=(
            selected_inputs, selected_output, finalfile, task, ID, x_inputs.columns.tolist(), db))

        t1.start()
        flash('The autoML is running... on an average take 2 - 3 minutes to run', 'info')
        return redirect(url_for('home'))


# def is_thread_alive(thread_id):
#     for t in threading.enumerate():
#         if t.ident == thread_id:
#             return t.is_alive()
#     return False


@app.route('/model_info/<id>', methods=['GET', 'POST'])
def model_info(id):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("select database();")
    record = cursor.fetchone()
    print(record)
    cursor.execute("SELECT * FROM automl.status WHERE ID = %s", (id,))
    print("current ID inside getdata", id)
    model_details = cursor.fetchall()

    with open('Files/model_' + str(id) + '/model_info_' + str(id) + '.json', "r") as file:
        model_data = json.load(file)
    cursor.close()
    return render_template('model_details.html', model_details=model_details, model_data=model_data)


@app.route('/predict/<id>', methods=['POST'])
def predict(id):
    print("inside predict")
    print(id)
    print(type(id))
    with open('Files/model_' + str(id) + '/model_info_' + str(id) + '.json', "r") as file:
        model_data = json.load(file)

    with open('Files/model_' + str(id) + '/field_info_' + str(id) + '.json', "r") as file:
        field_info = json.load(file)

    with open('Files/model_' + str(id) + '/metrics_info_' + str(id) + '.json', "r") as file:
        metrics = json.load(file)

    return render_template('prediction_3.html', model_data=model_data, field_info=field_info, metrics=metrics, id=id)

@app.route('/final_predict/<id>', methods=['POST'])
def final_predict(id):
    print("inside final predict")

    db = get_db()
    cursor = db.cursor()
    cursor.execute("select database();")
    record = cursor.fetchone()
    print(record)
    cursor.execute("SELECT Input_variables FROM automl.status WHERE ID = %s", (id,))
    print("current ID inside getdata", id)
    input_details = cursor.fetchone()
    cursor.close()
    print(input_details[0])
    print(type(input_details[0]))
    print(input_details[0].split(','))

    inputs2 = input_details[0].split(',')
    form_data = {}
    for i in inputs2:
        form_data[i] = request.form.get(i)
    print(form_data)
    print(pd.DataFrame(form_data, index=[0]))
    x2 = pd.DataFrame(form_data, index=[0])
    with open('Files/model_' + str(id) + '/model_' + str(id) + '.pkl', 'rb') as mod:
        loaded_model = pickle.load(mod)

    predy = loaded_model.predict(x2)
    print(predy[0])
    # with open('Files/model_' + str(id) + '/model_info_' + str(id) + '.json', "r") as file:
    #     model_data = json.load(file)

    # with open('Files/model_' + str(id) + '/metrics_info_' + str(id) + '.json', "r") as file:
    #     metrics = json.load(file)

    return jsonify(int(predy[0]))


if __name__ == '__main__':
    app.run(debug=True)
