# How to get running
1. install requirements 
```bash
pip install -r requirements.txt
```
2. unzip models and label_samples. Put the folders in the same directory as this README.md file.
```bash
tar -xzf models.tar.gz && tar -xzf label_samples.tar.gz
```
3. Change the `base_path` in [`config.py`](config.py) to the directory of this README.md file. 
I don't really know why i coded it like this, bit I'm not gonna change it now ¯\\\_(ツ)\_/¯.
