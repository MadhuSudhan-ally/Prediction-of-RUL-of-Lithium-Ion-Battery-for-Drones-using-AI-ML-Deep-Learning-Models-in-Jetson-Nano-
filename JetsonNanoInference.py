import numpy as np, tflite_runtime.interpreter as tflite
# load scaler (transfer scaler_raw.npz to device)
sc = np.load('scaler_raw.npz')
mean = sc['mean'].reshape(1,1,-1); std = sc['std'].reshape(1,1,-1)

interpreter = tflite.Interpreter('model_dynamic.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess(x): return np.clip((x - mean)/np.maximum(std,1e-6), -10, 10).astype('float32')

X = np.load('sample_window.npy')  # shape (1,50,13)
inp = preprocess(X)
interpreter.set_tensor(input_details[0]['index'], inp)
interpreter.invoke()
out = [interpreter.get_tensor(o['index']) for o in output_details]
print(out)