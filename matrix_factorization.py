import numpy as np


def array_split(X, split):
    ref = np.zeros_like(X.flatten())
    nonzero_idx = np.nonzero(X.flatten())[0]
    split_idx = np.random.choice(nonzero_idx, int(len(nonzero_idx) * split), replace=False)
    np.put(ref, split_idx, 1)
    ref = ref.reshape(X.shape)

    X_val = X * ref
    X_train = X - X_val
    return X_train, X_val


def build_train_val_test(file='mat_comp', val_split=0.15):
    with open(file) as f:
        num_users, num_movies, num_ratings = tuple(int(item) for item in (f.readline().strip()).split())
        M = np.zeros(shape=(num_users, num_movies))  # users x movies

        content = f.readlines()
        r = 0

        while r < num_ratings:
            i, j, rating = tuple(float(item) for item in (content[r].strip()).split())
            M[int(i - 1), int(j - 1)] = rating
            r += 1

        test_pos = np.zeros(shape=(num_users, num_movies))
        r += 1

        while r < len(content):
            i, j = tuple(int(item) for item in (content[r].strip()).split())
            test_pos[i - 1, j - 1] = 1
            r += 1

        train_pos = np.zeros(shape=(num_users, num_movies))
        val_pos = np.zeros(shape=(num_users, num_movies))

        M_train, M_val = array_split(M, val_split)
        return M, M_train, M_val, train_pos, val_pos, test_pos


def get_nonzero_samples(X):
    r, c = np.nonzero(X)
    vals = np.array(list(X[i, j] for i, j in zip(r, c)))
    return np.stack((r, c, vals), axis=1)


def get_batches(samples, batch_size):
    indices = np.asarray(list(range(0, len(samples), batch_size)) + [len(samples)])
    for start, end in zip(indices[:-1], indices[1:]):
        yield samples[start:end]


def get_mse(P, Q, M, valid=True):
    error_mat = P @ Q.T - M
    if valid:
        valid_mat = (M > 0).astype(np.int64)
    else:
        valid_mat = np.ones_like(M)

    mse = np.sum(np.square(error_mat * valid_mat)) / np.sum(valid_mat)
    return mse


def push_to_file(M, file='mat_comp_ans'):
    r, c = np.nonzero(M)
    lines = [0] * len(r)

    for k, i, j in zip(np.arange(len(r)), r, c):
        lines[k] = str(M[i, j])

    with open(file, 'w+') as f:
        f.write("\n".join(lines))


if __name__ == '__main__':
    # define hyperparameters
    k = 50
    alpha = 0.005
    beta = 0.02
    batch_sz = 500
    epochs = 10

    M, M_train, M_val, train_pos, val_pos, test_pos = build_train_val_test(file='mat_comp')

    P = np.random.normal(scale=1/k, size=(M.shape[0], k)).astype(np.float64)
    Q = np.random.normal(scale=1/k, size=(M.shape[1], k)).astype(np.float64)
    samples = get_nonzero_samples(M_train)

    for i in range(epochs):
        np.random.shuffle(samples)
        batch_gen = get_batches(samples, batch_sz)
        next_state = True

        while next_state:
            try:
                batch = next(batch_gen)
                user_idx = batch[:, 0].astype(int)
                movie_idx = batch[:, 1].astype(int)
                ratings = batch[:, 2]

                pred = np.diagonal(P[user_idx] @ Q[movie_idx].T)
                delta = (ratings - pred).astype(np.float64)
                P[user_idx] += alpha * (Q[movie_idx] * delta[:, np.newaxis] - beta * P[user_idx])
                Q[movie_idx] += alpha * (P[user_idx] * delta[:, np.newaxis] - beta * Q[movie_idx])

            except StopIteration:
                next_state = False

        train_mse = get_mse(P, Q, M_train)
        val_mse = get_mse(P, Q, M_val)
        print("Epoch {}, train MSE {:.4f}, validation MSE {:.4f}".format(i, train_mse, val_mse))

    val_score = 4 * (1.8 - val_mse)
    clipped_val_score = min(4 * max(0, (1.8 - val_mse)), 4)
    print("Validation score: {:.4f}; {}".format(val_score, clipped_val_score))

    M_test = (P @ Q.T) * (test_pos)
    push_to_file(M_test)
