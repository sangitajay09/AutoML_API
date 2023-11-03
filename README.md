Intern name - Sangita Jayendran


To execute AutoML app please do the following
---------------------------------------
- Create a database called automl in Xampp server and import the file "status.sql". This should set up the database and load it with the right entries.
- To execute AutoML app please use python Main_db.py


Files/folders related to the app
---------------------
- "Files" - contains folders of every model generated saved with the name "model_ID". It contains the following :-
    - Dataset uploaded by the cilent
    - "field_info_ID.json" - this contains the min and max values of each field.
    - "metrics_info_ID.json" - this contains all the evaluated scores and metrics of the model depending on its task.
    - "model_ID.pkl" - pickle folder of the fitted model.
    - "model_info_ID.json" - this contains other info about the tpot model.


- "Requiremnt.txt" - contains all the python modules to be downloaded.
- "Templates" - this contains all th html files.
- "static" - this contains css, js and bootstrap template files.
- "Dataset_used" - this contains the dataset used to create the 10 or 11 used cases.
- "Resources" - contains the "Industries_list.csv" used to populate th drop down menu for the domain values.
  
