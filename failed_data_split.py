import numpy as np

def split_data_inflate(chars_vector,
                       samples_per_batch,
                       sample_length,
                       split_frac=0.9):
    x = chars_vector[:-1]
    y = chars_vector[1:]

    sample_count = len(x) - sample_length + 1

    x_batches = []
    y_batches = []

    start_range = range(0, sample_count, sample_length)

    x_samples = np.array([x[start:start + sample_length] for start in start_range])
    y_samples = np.array([y[start:start + sample_length] for start in start_range])

    if sample_count > samples_per_batch:

        batch_count = len(x_samples) // samples_per_batch
        new_length = batch_count * samples_per_batch
        end_crop_count = len(x_samples) - new_length

        if end_crop_count != 0:
            x_samples = x_samples[:-end_crop_count]
            y_samples = y_samples[:-end_crop_count]

        x_batches = np.array(np.split(x_samples, batch_count))
        y_batches = np.array(np.split(y_samples, batch_count))

    else:

        x_batches = x_samples
        y_batches = y_samples

    split_idx = int(len(x_batches) * split_frac)

    train_x, train_y = x_batches[:split_idx], y_batches[:split_idx]
    val_x, val_y = x_batches[split_idx:], y_batches[split_idx:]

    return train_x, train_y, val_x, val_y


def get_batch_inflate(tx, ty):
    for x, y in zip(tx, ty):
        yield x, y