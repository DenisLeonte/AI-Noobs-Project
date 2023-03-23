import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import model_generation

def train(epochs):
    model_generation.train(epochs)

def res_train(epochs, last_epoch):
    model_generation.resume_training(epochs, last_epoch)

def plot(epochs):
    model_generation.plot_data(epochs)

#res_train(500, 53)
train(500)
#plot()