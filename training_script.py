'''Configurations for training'''
# Choose a batch size
BATCH_SIZE = 20

# Get the length of the training set
length_of_training_dataset = len(visualization_training_dataset)

# Get the length of the validation set
length_of_validation_dataset = len(visualization_validation_dataset)

# Get the steps per epoch (may be a few lines of code)
steps_per_epoch = math.ceil(length_of_training_dataset/BATCH_SIZE)

# get the validation steps (per epoch) (may be a few lines of code)
validation_steps = length_of_validation_dataset//BATCH_SIZE
if length_of_validation_dataset % BATCH_SIZE > 0:
    validation_steps += 1

'''Fit actually means training'''
history =  model.fit(get_training_dataset(visualization_training_dataset),
                    steps_per_epoch=steps_per_epoch, 
                    validation_data=get_validation_dataset(visualization_validation_dataset), 
                    validation_steps=validation_steps, 
                    epochs=EPOCHS)

'''A History object. Its History.history attribute is a record of training loss values and
metrics values at successive epochs, as well as validation loss values and 
validation metrics values (if applicable).'''


'''Validation Step'''
loss = model.evaluate(validation_dataset, steps=validation_steps)
print("Loss: ", loss)


'''Plotting Metrics'''
plot_metrics("loss", "Bounding Box Loss", ylim=0.2)
