##Demo of Image retrieval project in Computer Vision.

This is a simple demo of image retrieval based on pretrained CNN.

## Usage.

Please install requirements.txt first:

```
$ pip install requirements.txt
```


run the following command:

```
$ python image_retrieval_cnn.py
```

Your computer where the code run will work as a server, other terminals within the same LAN network can visit the website: "http://XXX.XXX.XXX.XXX:8080/", where "XXX.XXX.XXX.XXX" is ip of the server, type "ifconfig" in command widow to get it.

# Only Test.

If you only want to test the retrieval proccess, just read the code image_retrieval_cnn.py for reference, and run the following command:

```
$ cd retieval/
$ python retrieval.py
```

The sorted images will be printed.
