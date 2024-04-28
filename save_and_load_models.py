# Saving and Loading Models
# Save the model you just trained
model.save("temp_model.h5")

#Load the trained model
# Load the model you saved earlier
model = tf.keras.models.load_model("temp_model.h5", compile=False)