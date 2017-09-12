import csv
import numpy as np
import pickle
import matplotlib.pyplot as plt

def read_in_data(file_path):
    # initialize empty lists of values
    user_ids = []
    item_ids = []
    ratings = []
    timestamps = []
    
    with open(file_path) as file:
        reader = csv.reader(file)
        for row in reader:
            # each row is a list of length 1, tab separated values
            row_splits = row[0].split()

            # append values to lists
            user_ids.append(row_splits[0])
            item_ids.append(row_splits[1])
            ratings.append(row_splits[2])
            timestamps.append(row_splits[3])

    return (np.asarray(user_ids, dtype=int), np.asarray(item_ids, dtype=int)
            , np.asarray(ratings, dtype=int), np.asarray(timestamps, dtype=np.datetime64))

def create_matrix(data):
    '''
    data is a tuple of numpy arrays: user_id, item_id, rating, timestamp
    '''
    # parameters
    n_users = len(np.unique(data[0]))
    n_items = len(np.unique(data[1]))
    
    # put user, item, & rating into one array
    data = np.column_stack((data[0], data[1], data[2]))
    
##    # sort by item, then user
##    data = data[data[:,1].argsort()]
##    data = data[data[:,0].argsort(kind='mergesort')] # mergesort is a stable sort that keeps items with the same key in the same relative order. This is to break tie

    # create matrix
    matrix = np.empty((n_items, n_users), dtype=int)
    for user in range(n_users):
        user = user + 1 # start from 1 instead of 0

        for item in range(n_items):
            item = item + 1 # start from 1 instead of 0

            # get rating of user for item            
            try:
                rating = data[(data[:,0]==user) & (data[:,1]==item)][0,2]
            except IndexError:
                rating = 0

            # assign rating to its corresponding position
            matrix[item-1, user-1] = rating # start from 0 instead of 1

    return matrix

def save_data(data, saved_name):
    with open(saved_name + '.pickle', 'wb') as f:
        pickle.dump(data, f)

def load_data(saved_name):
    with open(saved_name + '.pickle', 'rb') as f:
        data = pickle.load(f)
    return data

def train(data_matrix, n_items, n_users, n_features, n_folds, alpha, reg_lambda, monitor='val_loss', seed=0):
    '''
    train model
    '''
	
    output_dict = {'features':[], 'thetas':[], 'train_losses':[], 'train_errors':[], 'val_losses':[], 'val_errors':[]}

    # split data into train & val sets
    for train_data, val_data in create_train_val_test_data(data_matrix, n_folds=n_folds):

        # initialize features & theta
        np.random.seed(seed)
        x = np.random.rand(n_items, n_features)
        theta = np.random.rand(n_users, n_features)

        # keep track of training statistics
        train_loss = []
        train_error = []
        val_loss = []
        val_error = []

        # run until condition met
        monitor_dict = {'val_loss': val_loss, 'val_error': val_error}
        while True:
            # calculate training loss
            loss = calculate_loss(x, theta, train_data, reg_lambda)
##            print('train loss:', loss)
            train_loss.append(loss)

            # calculate training error
            error = calculate_error(x, theta, train_data)
            train_error.append(error)

            # calculate gradient
            x_grad, theta_grad = calculate_gradient(x, theta, train_data, reg_lambda)

            # update weights
            x -= alpha*x_grad
            theta -= alpha*theta_grad

            # val loss
            loss = calculate_loss(x, theta, val_data, reg_lambda)
##            print('\n val loss:', loss)
            val_loss.append(loss)
##            print('previous val loss:', val_loss[-2])    

            # val error
            error = calculate_error(x, theta, val_data)
            val_error.append(error)

            # check if monitored value moved in the undesired direction: loss & error should be decreasing            
##            print(len(monitor_dict[monitor]))
            if len(monitor_dict[monitor]) > 1:                
                if (monitor_dict[monitor][-1] > monitor_dict[monitor][-2]):
                    # reverse weights
                    x += alpha*x_grad
                    theta += alpha*theta_grad

                    # remove last statistics
                    del train_loss[-1]
                    del train_error[-1]
                    del val_loss[-1]
                    del val_error[-1]
					
                    # save to output_dict
                    output_dict['features'].append(x)
                    output_dict['thetas'].append(theta)
                    output_dict['train_losses'].append(train_loss)
                    output_dict['train_errors'].append(train_error)
                    output_dict['val_losses'].append(val_loss)
                    output_dict['val_errors'].append(val_error)
					
                    # terminate training
                    break
                
    return output_dict

def create_train_val_test_data(data_matrix, test_size=None, n_folds=None):
    '''
    split data into train, val, & test sets
    test_size percentage of all data is used for testing
    only available ratings are used for testing & validation
    '''
    
    # parameters
    n_users = data_matrix.shape[1]
    
    # boolean matrix: same shape as data_matrix, with elements being 0 for no rating available, and 1 otherwise
    boolean_matrix = 1 * (data_matrix>0)
    
    # select indices of non-zero elements for each user.
    # result is a list of tuples, each element in list corresponds to each col of data matrix, each tuple is (list of 1st axis indices, list of 2nd axis indices)
    nonzero_indices = [np.where(boolean_matrix[:,user]>0) for user in range(n_users)]
    nonzero_indices_length = [len(i[0]) for i in nonzero_indices]

    # if splitting into train & val
    if test_size==None:
        test_size = 1

    # randomly select non-zero indices for the test set, ie., only available ratings are used for testing. 
    mask_sizes = [int(i * test_size) for i in nonzero_indices_length]
    np.random.seed(0)
    mask = [np.random.choice(i, size=j, replace=False) for i,j in zip(nonzero_indices_length, mask_sizes)]    

    # if splitting for test set
    if n_folds==None:
        n_folds = 1
        fold_sizes = mask_sizes
        left_overs = [0] * n_users
    else:
        # fold size
        fold_sizes = [(i // n_folds) for i in nonzero_indices_length]
        left_overs = [(i % n_folds) for i in nonzero_indices_length]    

    # set chunks of data to train/val & test sets
    starts = [0] * n_users
    for fold in range(n_folds):
        if fold == n_folds - 1:            
            ends = [(start + fold_size + left_over) for start, fold_size, left_over in zip(starts, fold_sizes, left_overs)]
        else:
            ends = [(start + fold_size) for start, fold_size in zip(starts, fold_sizes)]
        mask_fold = [i[start:end] for i, start, end in zip(mask, starts, ends)]

        # create train/val/test sets
        # train/val set
        b_train = boolean_matrix.copy()
        for user in range(n_users):
            b_train[(nonzero_indices[user][0][mask_fold[user]],user)] = 0

        # test_set
        b_test = boolean_matrix - b_train 

        yield (data_matrix*b_train, data_matrix*b_test)
        starts = ends

def calculate_loss(x, theta, y, reg_lambda):
    '''
    calculate loss
    x: (n_items, n_features)
    theta: (n_users, n_features)    
    y: (n_items, n_users), user-ratings matrix
    '''
    # boolean matrix: same shape as y, with elements being 0 for no rating available, and 1 otherwise
    b = 1 * (y>0)
    
    # loss
    loss = 1/2 * (((np.dot(x, theta.transpose()) - y) * b) ** 2).sum() + reg_lambda/2 * (x**2).sum() + reg_lambda/2 * (theta**2).sum()
    
    return loss

def calculate_gradient(x, theta, y, reg_lambda):
    '''
    calculate gradient
    x: (n_items, n_features)
    theta: (n_users, n_features)    
    y: (n_items, n_users), user-ratings matrix
    '''
    # boolean matrix: same shape as y, with elements being 0 for no rating available, and 1 otherwise
    b = 1 * (y>0)

    # gradient                
    x_grad = np.dot(((np.dot(x, theta.transpose()) - y) * b), theta) + reg_lambda*x   
    theta_grad = np.dot(((np.dot(x, theta.transpose()) - y) * b).transpose(), x) + reg_lambda*theta
    
    return x_grad, theta_grad

def calculate_error(x, theta, y):
    '''
    calculate rmse
    x: (n_items, n_features)
    theta: (n_users, n_features)    
    y: (n_items, n_users), user-ratings matrix
    '''
    # boolean matrix: same shape as y, with elements being 0 for no rating available, and 1 otherwise
    b = 1 * (y>0)

    # prediction & error
    y_pred = np.dot(x, theta.transpose()) * b
    error = np.sqrt(((y_pred - y) * b)**2).sum()/b.sum()
    
    return error

def visualize_training(output_dict):    
    n_folds = len(output_dict['train_losses'])
    
    for i in range(n_folds):
        # losses
        plt.figure(figsize=(16, 2*n_folds))
        plt.subplot(n_folds,2,i*2+1)
        plt.plot(output_dict['train_losses'][i])
        plt.plot(output_dict['val_losses'][i])
        plt.ylabel('loss')
        plt.xlabel('iteration')
        plt.legend(['train loss', 'val loss'], loc='upper left')
        
        # errors
        plt.subplot(n_folds,2,i*2+2)
        plt.plot(output_dict['train_errors'][i])
        plt.plot(output_dict['val_errors'][i])
        plt.ylabel('error')
        plt.xlabel('iteration')
        plt.legend(['train error', 'val error'], loc='upper left')

        plt.tight_layout()

def predict_ensemble(xs, thetas, y):
    '''
    predict ratings for each model, then average predictions for final output
    '''

    # boolean matrix: same shape as y, with elements being 0 for no rating available, and 1 otherwise
    b = 1 * (y>0)

    # parameters
    n_models = len(xs)

    # predict & error for each model
    y_preds = []
    errors = []
    for i in range(n_models):
        y_pred = np.dot(xs[i], thetas[i].transpose()) * b
        y_preds.append(y_pred)

        # error
        error = np.sqrt(((y_pred - y) * b)**2).sum()/b.sum()
        errors.append(error)

    # emsemble statistics
    y_pred_ensemble = np.asarray(y_preds).mean(axis=0)
    error_ensemble = np.sqrt(((y_pred_ensemble - y) * b)**2).sum()/b.sum()

    return {'y_preds':y_preds, 'y_pred_ensemble':y_pred_ensemble, 'errors':errors, 'error_ensemble':error_ensemble}
