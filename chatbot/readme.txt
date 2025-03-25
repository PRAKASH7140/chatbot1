python -m venv venv
venv\Scripts\Activate
python -m pip install --upgrade pip
pip install -r requirements.txt 

pip install --upgrade pip setuptools wheel
pip install numpy==1.24.4
pip install spacy==3.2.4
pip install -r requirements.txt

python -m spacy download en_core_web_sm

python app.py

pip install ultralytics
pip install  mediapipe
Remove-Item -Recurse -Force venv 