

from fashion_mnist_preprocess import (load_mnist_dataset,
                                      shuffle_training_data,
                                      flatten_images,
                                      create_fashion_mnist_model,
                                      add_prj_root_dir_to_path)


def run():
    X, y = load_mnist_dataset('train', 'fashion_mnist_images')
    X_valid, y_valid = load_mnist_dataset('test', 'fashion_mnist_images')
    X, y = shuffle_training_data(X, y)
    X = flatten_images(X)
    X_valid = flatten_images(X_valid)
    create_fashion_mnist_model(X, y, X_valid, y_valid)



if __name__ == '__main__':
    add_prj_root_dir_to_path()
    run()