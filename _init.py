from Model import Model
from models.Model_Fine_Tuning import Model_FT_ResNet50
from models.Model_CNN import Model_CNN
import visualkeras

if __name__ == "__main__":

    # --------------- Train model fine tuning ResNet50 ---------------- #
    # model_FT = Model_FT_ResNet50(model_name = "FT_ResNet50_5_layers-50-10", nb_layers_to_learn=5)
    # visualkeras.layered_view(model_FT._model, to_file='results/Architecture_model_ResNet50.png', legend=True)
    # model_FT.train(epochs=50,patience=10)
    # model_FT.save()

    # --------------- Evaluate model fine tuning ResNet50 ---------------- #
    # model_FT.load(path="saved_models/FT_ResNet50_5_layers-50-10.h5")
    # model_FT.evaluate()

    # --------------- Train model CNN from scratch ---------------- #
    model_CNN = Model_CNN(model_name = "CNN-50-10")
    model_CNN.train(epochs=50,patience=10)
    model_CNN.save()

    # --------------- Evaluate model fine tuning ResNet50 ---------------- #
    # model_CNN = Model_CNN(model_name = "CNN")
    # model_CNN.load()
    # visualkeras.layered_view(model_CNN._model, to_file='results/visualisation/Architecture_model_CNN.png', legend=True)
    # model_CNN.evaluate()
    # model_CNN.infos_model()

