function p2m()
%change Pickle file to mat file
    pyscript = ['import cPickle;import gzip;import numpy, scipy.io;f = gzip.open("mnist.pkl.gz", "rb");[train_set, valid_set, test_set] = cPickle.load(f);f.close();train_x, train_y = train_set;valid_x, valid_y = valid_set;test_x, test_y = test_set;scipy.io.savemat("./mat_data_train_x.mat", mdict={"train_x": train_x});scipy.io.savemat("./mat_data_train_y.mat", mdict={"train_y": train_y});scipy.io.savemat("./mat_data_valid_x.mat", mdict={"valid_x": valid_x});scipy.io.savemat("./mat_data_valid_y.mat", mdict={"valid_y": valid_y});scipy.io.savemat("./mat_data_test_x.mat", mdict={"test_x": test_x});scipy.io.savemat("./mat_data_test_y.mat", mdict={"test_y": test_y})'];
    system(['python -c ''' pyscript '''']);
end