import os
import shutil
import re
from .pretrained_model import ImageFineModel, my_call_backs
from xl_tool.xl_io import read_json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Nadam, Adadelta, Adamax, Adagrad, Ftrl
import numpy as np
import matplotlib.pyplot as plt
import logging
from ..preprocessing.tfdata import image_from_tfrecord

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s',
                    level=logging.INFO,
                    handlers=[logging.FileHandler("./training_log"), logging.StreamHandler()])

plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
eff_input_dict = {'efficientnetb0': 224, 'efficientnetb1': 240,
                  'efficientnetb2': 260,
                  'efficientnetb3': 300,
                  'efficientnetb4': 380}

optimizer_dict = {
    "RMSprop": RMSprop,
    "Adam": Adam,
    "Ftrl": Ftrl,
    "SGD": SGD,
    "Nadam": Nadam,
    "Adamax": Adamax,
    "Adadelta": Adadelta, "Adagrad": Adagrad

}


def file_scanning(path, file_format=r".txt$", full_path=True, sub_scan=False):
    """
        scanning directory and return file paths with specified format
        :param path: directory to scan
        :param file_format:  file format to return ,regular patterns
        :param full_path: whether to return the full path
        :param sub_scan: whether to sanning the subfolder
        :return:file paths
        """
    if os.path.exists(path):
        file_paths = []
        for root, dirs, files in os.walk(path, topdown=True):
            paths = [file for file in files if re.search(file_format, file)]
            if full_path:
                paths = [os.path.join(root, file) for file in paths]
            file_paths.extend(paths)
            if not sub_scan:
                break
        if not file_paths:
            print("File with specified format not find")
            return
    else:
        print("Invalid path!")
        return
    return file_paths


def data_gen_from_one(target_size=(224, 224), batch_size=10):
    train_path = r"E:\foodDetection_5_classes_first_20191227_train"
    datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=20, width_shift_range=0.1,
                                 height_shift_range=0.1, zoom_range=0.1, validation_split=0.2)
    train_gen = datagen.flow_from_directory(train_path, target_size=target_size, batch_size=batch_size,
                                            subset="training")
    val_gen = datagen.flow_from_directory(train_path, target_size=target_size, batch_size=batch_size,
                                          subset="validation")
    return train_gen, val_gen


def train_data_from_directory(train_path, val_path, target_size=(224, 224), batch_size=16,
                              rescale=1. / 255, rotation_range=20, width_shift_range=0.2,
                              height_shift_range=0.20, zoom_range=0.3, vertical_flip=True,
                              horizontal_flip=True, brightness_range=(0.7, 1.2), classes=None):
    """从指定数据集生成数据，如果没有验证集请将val_path设置为空"""
    train_datagen = ImageDataGenerator(rescale=rescale, rotation_range=rotation_range,
                                       width_shift_range=width_shift_range,
                                       height_shift_range=height_shift_range, brightness_range=brightness_range,
                                       zoom_range=zoom_range, vertical_flip=vertical_flip,
                                       horizontal_flip=horizontal_flip)
    val_datagen = ImageDataGenerator(rescale=rescale)
    train_gen = train_datagen.flow_from_directory(train_path, classes=classes, target_size=target_size,
                                                  batch_size=batch_size)
    if val_path:
        val_gen = val_datagen.flow_from_directory(val_path, target_size=target_size, classes=classes,
                                                  batch_size=batch_size)
        if train_gen.class_indices == val_gen.class_indices:
            return train_gen, val_gen

        else:
            logging.info("训练集与验证集类别定义不一致！")
            return False
    else:
        return train_gen


def finetune_model(name="", prefix="", class_num=6, train_path="./dataset/specified_scenario/train",
                   val_path="./dataset/specified_scenario/val", tf_record=False, tf_record_label2id=None,
                   weights="imagenet", train_from_scratch=False, patience=6, initial_epoch=0, dropout=False,
                   test=True, classes=None, epochs=(5, 30, 60, 120), lrs=(0.00001, 0.003, 0.0003, 0.00003),
                   optimizer="adam",reducelr=3,
                   batch_size=16, target_size=(224, 224), train_buffer_size=5000, val_buffer_size=5000, prefetch=False):
    """预训训练最后一层与全部训练对比"""
    if tf_record:
        if prefetch:
            train_gen = image_from_tfrecord(train_path, class_num, batch_size,
                                            target_size=target_size, random_brightness=0.15,
                                            random_contrast=(0.8, 1.3), rotate=0.3, zoom_range=0.15, noise=0.04,
                                            random_crop=0.85, random_flip_left_right=True, random_flip_up_down=False,
                                            random_aspect=True, buffer_size=train_buffer_size).prefetch(
                tf.data.experimental.AUTOTUNE)
            val_gen = image_from_tfrecord(val_path, class_num, batch_size,
                                          target_size=target_size, buffer_size=val_buffer_size).prefetch(
                tf.data.experimental.AUTOTUNE)
        else:
            train_gen = image_from_tfrecord(train_path, class_num, batch_size,
                                            target_size=target_size, random_brightness=0.15,
                                            random_contrast=(0.8, 1.3), rotate=0.3, zoom_range=0.15, noise=0.04,
                                            random_crop=0.85, random_flip_left_right=True, random_flip_up_down=False,
                                            random_aspect=True, buffer_size=train_buffer_size)
            val_gen = image_from_tfrecord(val_path, class_num, batch_size,
                                          target_size=target_size, buffer_size=val_buffer_size)
        cat_id = read_json(tf_record_label2id)
        print(cat_id)
        labels = [j[1] for j in sorted([(i[1], i[0]) for i in cat_id.items()], key=lambda x: x[1])]
        with open(f"./model/labels/{prefix + name + f'_{class_num}_labels.txt'}", "w") as f:
            f.write("\n".join(labels))
    else:
        if type(classes) == str:
            print(classes)
            classes = read_json(classes)
            classes = [i[0] for i in sorted(classes.items(), key=lambda x: x[1])]
        train_gen, val_gen = train_data_from_directory(train_path, val_path, classes=classes, batch_size=batch_size,
                                                       rescale=1. / 255, rotation_range=20, width_shift_range=0.2,
                                                       height_shift_range=0.20, zoom_range=0.3, vertical_flip=True,
                                                       horizontal_flip=True, brightness_range=(0.7, 1.2))
        cat_id = dict()
        cat_id['cat2id'] = (train_gen.class_indices)
        print(cat_id['cat2id'])
        labels = [i[0] for i in sorted(train_gen.class_indices.items(), key=lambda x: x[1])]
        with open(f"./model/labels/{prefix + name + f'_{class_num}_labels.txt'}", "w") as f:
            f.write("\n".join(labels))
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            # indexs = []
    if not train_from_scratch:
        strategy = tf.distribute.MirroredStrategy()
        logging.info('Number of devices: %d' % strategy.num_replicas_in_sync)
        with strategy.scope():
            model = ImageFineModel.create_fine_model(name, class_num, weights=weights, prefix=prefix,
                                                     suffix=f"_{class_num}", dropout=dropout,
                                                     non_flatten_trainable=True, input_shape=(*target_size, 3))
            call_back = my_call_backs(model.name, patience=patience,reducelr=reducelr)
        if test:
            model.fit(train_gen, validation_data=val_gen, epochs=2, callbacks=call_back, steps_per_epoch=2,
                      validation_steps=2, use_multiprocessing=True, workers=5)
        else:
            with strategy.scope():
                model.compile(optimizer_dict[optimizer](lrs[0]), loss="categorical_crossentropy",
                              metrics=list(["accuracy", ]))
            model.fit(train_gen, validation_data=val_gen, epochs=epochs[0], callbacks=call_back,
                      initial_epoch=0, use_multiprocessing=False)
            with strategy.scope():
                model.compile(optimizer_dict[optimizer](lrs[1]), loss="categorical_crossentropy",
                              metrics=list(["accuracy", ]))
            model.fit_generator(train_gen, validation_data=val_gen, epochs=epochs[1], callbacks=call_back,
                                initial_epoch=epochs[0], use_multiprocessing=False)
    else:
        print("从头开始训练模型！！！")
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = ImageFineModel.create_fine_model(name, class_num,
                                                     weights=weights if weights == "imagenet" else None, prefix=prefix,
                                                     suffix=f"_{class_num}", dropout=dropout,
                                                     non_flatten_trainable=True, input_shape=(*target_size, 3))
        if weights and weights != "imagenet":
            model.load_weights(weights)
        call_back = my_call_backs(model.name, patience=patience,reducelr=reducelr)
        if test:
            model.fit_generator(train_gen, validation_data=val_gen, epochs=2, callbacks=call_back, steps_per_epoch=2,
                                validation_steps=2, use_multiprocessing=True, workers=5)
        else:
            for (i, epoch) in enumerate(epochs):
                if initial_epoch >= epoch:
                    continue
                with strategy.scope():
                    model.compile(optimizer_dict[optimizer](lrs[i]), loss="categorical_crossentropy",
                                  metrics=list(["accuracy", ]))
                model.fit(train_gen, validation_data=val_gen, epochs=epoch,
                          callbacks=call_back,
                          initial_epoch=initial_epoch, use_multiprocessing=False)
                initial_epoch = epochs[i]


def visual_misclassified_images(base_model, cat_num, weights, dataset, save_path,
                                target_size=(224, 224), batch_size=32, test=False, classes=None):
    root_dir = os.path.split(os.path.abspath(dataset))[1]
    model = ImageFineModel.create_fine_model(base_model, cat_num=cat_num, weights=None)
    model.load_weights(weights)
    os.makedirs(save_path, exist_ok=True)
    target_size = target_size if base_model not in eff_input_dict.keys() else (
        eff_input_dict[base_model], eff_input_dict[base_model],)
    if type(classes) == str:
        classes = read_json(classes)
        classes = [i[0] for i in sorted(classes.items(), key=lambda x: x[1])]
    gen = ImageDataGenerator(rescale=1.0 / 255.0).flow_from_directory(
        dataset, shuffle=False, target_size=target_size, batch_size=batch_size, classes=classes)
    # print("类别排序：", "\n".join(classes), gen.class_indices)
    filenames = gen.filenames
    classes = gen.classes
    cat2id = gen.class_indices
    id2cat = dict([(i[1], i[0]) for i in cat2id.items()])
    predict_p = model.predict_generator(gen)
    predict_classes = predict_p.argmax(-1)
    # arg_sort = predict_p.argsort(axis=-1)
    count = 1
    for i in range(len(filenames)):
        if classes[i] == predict_classes[i]:
            continue
        else:
            if test:
                print("发现误分类样本：", filenames[i])
            true_label = id2cat[classes[i]]
            false_label = id2cat[predict_classes[i]]
            mis_filename = f"{true_label}__{false_label}__{root_dir}_{count}.jpg"
            shutil.copy(f"{dataset}/{filenames[i]}", f"{save_path}/{mis_filename}")
            count += 1
    cat_acc1, cat_acc2, top1, top2 = mul_classify_acc(predict_p, classes, cat_num)
    s = (f">>>top1类别准确率：{top1}\n") + (f">>>top2类别准确率：{top2}\n") + \
        (">>>单类别类准确率：\ntop1\t\ttop2\n" + "\n".join([id2cat[i].rjust(10, " ") + ":\t" + str(cat_acc1[i])[:4]
                                                    + "\t" + str(cat_acc2[i])[:4] for i in range(cat_num)]))
    print(s)
    with open(f"{save_path}/mis_classify_result.txt", "w", encoding="utf-8") as f:
        f.write(s)


def mul_classify_acc(predict_p, real, cat_num):
    unique = list(range(cat_num))
    arg_sort = predict_p.argsort(axis=-1)

    pred = arg_sort[:, -1]
    print(sum((predict_p.argmax(axis=-1) == pred)))
    count = len(pred)
    pred = np.array(pred)
    real = np.array(real)
    all_eval = (pred == real)
    all_eval_2 = (arg_sort[:, -2] == real)
    cat_acc1 = [sum((real == i) & all_eval) / sum((real == i)) for i in unique]
    cat_acc2 = [(sum(((real == i) & (all_eval_2 | all_eval)))) / sum((real == i)) for i in unique]
    top1 = sum(all_eval) / count
    top2 = (sum(all_eval_2) / count) + top1
    return cat_acc1, cat_acc2, top1, top2
