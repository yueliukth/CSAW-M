import data_handler
import helper

SPECIALS = {
    # 'skip': ['test_28.png', 'test_65.png', 'test_107.png', 'test_120.png', 'test_143.png', 'test_151.png', 'test_158.png',
    #          'test_187.png', 'test_192.png', 'test_206.png', 'test_214.png', 'test_218.png', 'test_232.png', 'test_257.png', 'test_268.png',
    #          'test_273.png', 'test_290.png', 'test_318.png', 'test_385.png', 'test_400.png', 'test_427.png', 'test_432.png',
    #          'test_457.png', 'test_467.png']
    'skip': {
        'filenames': ['test_192.png', 'private_160.png', 'private_328.png', 'private_442.png'] +
                     ['train_849.png', 'train_1106.png', 'train_1303.png', 'train_1318.png', 'train_1392.png', 'train_1536.png',
                      'train_1830.png', 'train_1859.png', 'train_2087.png', 'train_2156.png', 'train_2191.png', 'train_2498.png',
                      'train_3535.png', 'train_3924.png', 'train_4314.png', 'train_4873.png', 'train_5033.png', 'train_5043.png',
                      'train_5373.png', 'train_5925.png', 'train_5931.png', 'train_6161.png', 'train_6349.png', 'train_6792.png',
                      'train_6826.png', 'train_7083.png', 'train_7154.png', 'train_7311.png', 'train_7540.png', 'train_7738.png',
                      'train_7861.png', 'train_7869.png', 'train_8081.png', 'train_8117.png', 'train_8486.png', 'train_8502.png',
                      'train_8526.png', 'train_8598.png', 'train_8636.png', 'train_8730.png', 'train_8794.png', 'train_9307.png']
    },
    'special_1': {
        'filenames': ['private_187.png', 'private_251.png', 'private_340.png'] +
                     ['train_624.png', 'train_741.png', 'train_1004.png', 'train_1267.png', 'train_1349.png', 'train_1641.png',
                      'train_2155.png', 'train_3014.png', 'train_3595.png', 'train_3704.png', 'train_4687.png',
                      'train_4732.png', 'train_6536.png'],
        'config': {'kernel_size': (15, 15)}
    },
    'special_2': {
        'filenames': ['private_415.png'] +
                     ['train_833.png', 'train_2073.png', 'train_3885.png', 'train_7599.png', 'train_7719.png'],
        'config': {'min_contour_area': 200}
    },
    'specials_3': {
        'filenames': ['train_797.png'],
        'config': {'max_contour_area': 500}
    }

}


def generate_preprocessed_images(raw_folder, preprocessed_folder, labels_path):
    n_processes = 20
    ignore_non_specials = False
    data_handler.raw_to_preprocessed(raw_folder=raw_folder,
                                     labels_path=labels_path,
                                     save_dir=preprocessed_folder,
                                     specials=SPECIALS,
                                     ignore_non_specials=ignore_non_specials,
                                     n_processes=n_processes)

