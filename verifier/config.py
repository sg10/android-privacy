data_folder = "/data"


class Samples:
    # contains subfolders (categories)
    app_metadata = data_folder + "/samples/metadata/"

    samples_database = data_folder + "/samples/samples_database.pk"
    list_of_cross_platform_apps = data_folder + "/samples/cross_platform_apps.txt"

    test_set_save_file = data_folder + "/test_apps.txt"

    # 50% most downloaded apps, 50% least downloaded apps
    # only english
    num_for_test_set = 1000


class Clustering:
    min_downloads_visualize = 4e6


class Text2PermissionClassifier:
    batch_size = 32
    max_train_epochs = 300
    validation_split = 6

    early_stopping_patience = 16
    early_stopping_delta = 0.02  # 2%

    max_description_embeddings = 600
    embedding_dim = 301  # +1 for flags
    #downloaded_embedding_file = data_folder + "/word_embeddings/word2vec-wiki-news-300d-1M.vec"
    downloaded_embedding_file = data_folder + "/word_embeddings/glove.6B.300d.txt"
    cached_embedding_file_indices = data_folder + "/word_embeddings/cached-indices.pk"
    cached_embedding_file_values = data_folder + "/word_embeddings/cached-values.np"
    cached_embedding_file_check = data_folder + "/word_embeddings/cache-check.txt"
    pre_embedded_samples_indices = data_folder + "/word_embeddings/samples-indices.pk"

    test_set_lime = data_folder + '/lime/t2p.json'

    conv_filters_num = 1024
    conv_filters_sizes = [1, 2, 3]
    dense_layers = [5000, 2500]
    dropout = 0.2

    heatmap_threshold = 0.49


class Permissions:
    groups_list = data_folder + "/permission_groups.json"


class TrainedModels:
    models_dir = data_folder + "/trained_models/"
    text2permission = models_dir + "text2permission.h5"
