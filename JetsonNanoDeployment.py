
#!pip install -q tf2onnx
python - <<'PY'
import tf2onnx, tensorflow as tf
model = tf.keras.models.load_model('/path/ens_0.keras', compile=False)
spec = (tf.TensorSpec((None,50,13), tf.float32, name="input"),)
output_path = "/path/model_ens0.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
print("Saved ONNX:", output_path)
PY

#____________________________
# convert best ensemble model manually (choose one ens_i.keras)
#python - <<'PY'
from tensorflow.keras.models import load_model
import tensorflow as tf
m = load_model('/path/ens_0.keras', compile=False)
converter = tf.lite.TFLiteConverter.from_keras_model(m)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tfl = converter.convert()
open('/path/model_dynamic.tflite','wb').write(tfl)
print('Saved tflite')
#PY



