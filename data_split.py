from utils.utils import load_data, split_dataset, save_seqs
import argparse

def main():
    global writer, best_epoch
    # Training settings
    parser = argparse.ArgumentParser(description='Official version of iBindNet')
    # Data options
    parser.add_argument('--data_dir',       type=str, default="data", help='data path')
    parser.add_argument('--prefix',       type=str, default="prefix", help='the prefix of data')
    args = parser.parse_args()

    filename = args.data_dir + "/" + args.prefix + ".txt"
    print("loading data from "+ filename) 
    sequences,targets = load_data(filename)
    train, test = split_dataset(sequences, targets, valid_frac=0.2)

    X_train=train[0]
    y_train=train[1]
    X_test=test[0]
    y_test=test[1]


    train_fn = args.data_dir + "/" + args.prefix + "_train.txt"
    print("saving training data into "+train_fn)
    save_seqs(X_train,y_train,train_fn)

    test_fn = args.data_dir + "/" + args.prefix + "_test.txt"
    print("saving test data into "+test_fn)
    save_seqs(X_test,y_test,test_fn)

    
    
if __name__ == '__main__':
    main()