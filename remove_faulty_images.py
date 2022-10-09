import os
import tensorflow as tf

to_delete = []

for dir in os.listdir("data"):
    for filename in os.listdir(f"data/{dir}"):
        f = open(f"data/{dir}/{filename}", "rb")
        try:
            tf.io.decode_image(f.read())
        except:
            to_delete.append(f"data/{dir}/{filename}")
        finally:
            f.close()

for file in to_delete:
    os.remove(file)