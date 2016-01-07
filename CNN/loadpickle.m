function [t_x,t_y,v_x,v_y,tt_x,tt_y] = loadpickle()
    %change Pickle file to mat file
    p2m();
    
    %read data
    t_x = load('mat_data_train_x.mat');
    t_y = load('mat_data_train_y.mat');
    v_x = load('mat_data_valid_x.mat');
    v_y = load('mat_data_valid_y.mat');
    tt_x = load('mat_data_test_x.mat');
    tt_y = load('mat_data_test_y.mat');
    
    t_x = t_x.train_x;
    t_y = t_y.train_y';
    v_x = v_x.valid_x;
    v_y = v_y.valid_y';
    tt_x = tt_x.test_x;
    tt_y = tt_y.test_y';
end