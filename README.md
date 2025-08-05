# How to get running
1. install requirements 
```bash
pip install -r requirements.txt
```
 Installing `torch` and `torchvison` from the requirements file might not work. You might have to install them manually from these repositories: 
 - [torch](https://download.pytorch.org/whl/torch/) version 2.3.1
 - [torchvision](https://download.pytorch.org/whl/torchvision/) version 0.18.1

 Look for the package that matches your python version and operating system. For example, `torch-2.3.1+cpu-cp312-cp312-linux_x86_64.whl` is torch version 2.3.1 with cpu processing for python 3.12 on linux x86_64 architecture. You can Install these packages with pip like this:
```
pip install https://download.pytorch.org/whl/cpu/torch-2.3.1%2Bcpu-cp312-cp312-linux_x86_64.whl
```
2. unzip models and label_samples. Put the folders in the same directory as this README.md file.
```bash
mkdir models label_samples
tar -xzf models.tar.gz -C models && tar -xzf label_samples.tar.gz -C label_samples
```

If you have this behind a reverse proxy, you can change the `base_url_path` variable in `app.py` to the path you want to use. For example, if you want to access the app at `https://example.com/landmark`, set `base_url_path = "/landmark"`.
You can also set the `BASE_URL_PATH` environment variable to the same value, which will be used by the app.

# Run using Docker

This app is available as a Docker image.
Run it directly from the command line:
```bash
docker run -d -p 5025:5025 -e BASE_URL_PATH="/landmark" cedrikewers/landmark:latest
```
or use docker-compose:
```yaml
version: '3.8'
services:
    landmark:
        image: cedrikewers/landmark:latest
        ports:
        - "5025:5025"
        environment:
        - BASE_URL_PATH: "/landmark" # remove this if you're not using a reverse proxy
```

The app will be available at `http://127.0.0.1:5025`.