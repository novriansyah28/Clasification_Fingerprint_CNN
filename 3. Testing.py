import os
import time
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model

img_width, img_height = 128, 128
model_path = './models/model(epoch=100,lr=0.001,Op=adam)/model.h5'
model_weights_path = './models/model(epoch=100,lr=0.001,Op=adam)/weights.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)

start= time.time()

def predict(file):
    x = load_img(file, target_size=(img_width,img_height))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = model.predict(x)
    result = array[0]
    answer = np.argmax(result)

    if answer == 0:
        print("Label: Arch")
    elif answer == 1:
        print("Label: Loop")
    elif answer == 2:
        print("Label: Whorl")
    
    return answer

arch_t = 0
arch_f = 0
loop_t = 0
loop_f = 0
whorl_t = 0
whorl_f = 0
for i, ret in enumerate(os.walk('./data/test/arch')):
    for i, filename in enumerate(ret[2]):
        if filename.startswith("."):
            continue
        print("Label: arch")
        result = predict(ret[0] + '/' + filename)
        if result == 0:
            arch_t += 1
        else:
            arch_f += 1
            
for i, ret in enumerate(os.walk('./data/test/loop')):
    for i, filename in enumerate(ret[2]):
        if filename.startswith("."):
            continue
            print("Label: loop")
            result = predict(ret[0] + '/' + filename)
        if result == 1:
            loop_t += 1
        else:
            loop_f += 1

for i, ret in enumerate(os.walk('./data/test/whorl')):
    for i, filename in enumerate(ret[2]):
        if filename.startswith("."):
            continue
        print("Label: whorl")
        result = predict(ret[0] + '/' + filename)
        if result == 2:
            whorl_t += 1
        else:
            whorl_f += 1
""" 
Check metrics 
""" 
print("True Arch: ", arch_t)
print("False Arch: ", arch_f)
print("True Loop: ", loop_t)
print("False Loop: ", loop_f)
print("True Whorl: ", whorl_t)
print("False Whorl: ", whorl_f)

end = time.time()
dur = end-start
if dur<60:
    print("Execution Time: ", dur," Seconds")
elif dur>0 and dur<3600:
    dur=dur/60
    print("Execution Time: ", dur," Minutes")
else:
    dur=dur/(60*60)
    print("Execution Time: ", dur," Hours")
