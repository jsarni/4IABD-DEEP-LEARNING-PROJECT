from tensorflow.keras.callbacks import TensorBoard
from cifar10.models.Models import getRandomModelID
from datetime import date


def test_models(model_type, models, models_descriptions, train_ds, train_labels, test_ds, test_labels, epochs_p, batch_size_p=4096):
    cur_date = date.today().strftime("%Y%m%d")
    logs_dir = '.\\logs\\{}\\{}\\'.format(model_type, cur_date)
    fit_models_dir = '.\\trained_models\\{}\\{}\\'.format(model_type, cur_date)
    trained_models = 0

    for e in epochs_p:
        for i in range(len(models)):
            print("********************************************* Training model number {} *********************************************".format(e*i + i + 1))
            model_id = getRandomModelID()
            log_name = logs_dir + model_type + '_' + model_id + '.log'
            model_name = fit_models_dir + model_type + '_' + model_id + '.h5'
            tensorboard_callback = TensorBoard(log_dir=log_name)

            models[i].fit(train_ds,
                          train_labels,
                          validation_data=(test_ds, test_labels),
                          epochs=e,
                          batch_size=batch_size_p,
                          callbacks=[tensorboard_callback]
                          )
            # models[i].save(model_name)

            model_descr = "{};{};{}\n".format(models_descriptions[i], str(e), model_id)
            with open(".\\trained_models\\mlp\\tested_{}_history.csv".format(model_type, model_type), "a") as f:
                f.write(model_descr)

            trained_models += 1

    return trained_models


