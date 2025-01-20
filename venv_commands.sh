
# create venv
python3.10 -m venv venv


# install dependencies
source venv/bin/activate
pip install pandas torch PyYAML
pip install -e .
