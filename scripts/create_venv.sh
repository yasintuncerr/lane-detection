if [ -d "deeplr_venv" ]; then
  rm -rf deeple_venv
fi

cd ..
python3 -m venv deeplr_venv
source deeplr_venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
