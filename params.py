# params for baseline

dataset_dir = {
    'amazon':'./data/amazon/images',
    'webcam':'./data/webcam/images',
    'dslr':'./data/dslr/images'
}

batch_size = 64
image_size = 227
mean = [0.485,0.456,0.406]
std = [0.229,0.224,0.225]
shuffle = True
num_workers = 2


num_epoch = 200

logfile_name = 'log.log'
model_save_name = 'dpg.pkl'





