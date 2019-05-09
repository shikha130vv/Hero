# Hero
# Download all code from github and unzip in a folder say Hero-master
# Create a new environment If using conda, then use 
conda create -n Heroenv 
# Activate the environment. On my ubuntu system command is 
source activate Heroenv
# Go to the directory where we unzipped the code: 
cd Hero-matser
# install requirements
pip install -r requirements.txt
# create init files
cd posts
touch __init__.py
cd migrations
touch __init__.py
cd ..
cd ..
# Make migrations
python manage.py makemigrations
# Migrate
python manage.py migrate
# start server
python manage.py runserver
# browse to site
http://127.0.0.1:8000

