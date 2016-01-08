clear all;

% read data
[train_x,train_y,valid_x,valid_y,test_x,test_y] = loadpickle();

learning_rate = 0.1;
n_epochs = 200;
nkerns = [20 50];
batch_size = 500;

n_train_batches = length(train_x);
n_valid_batches = length(valid_x);
n_test_batches = length(test_x);
n_train_batches = n_train_batches / batch_size;
n_valid_batches = n_valid_batches / batch_size;
n_test_batches = n_test_batches / batch_size;

patience = 10000;
patience_increase = 2;

improvement_threshold = 0.995;

validation_frequency = min(n_train_batches,patience/2);

best_validation_loss = 999999;
best_iter = 0;
test_score = 0;

looping = 1;
epoch = 0;

%Build Layers
fprintf('Building...\n');

% hiddenlayer - tanh
n_in = 28*28;
n_out = 500;
[w0,b0] = hiddenlayer_build(n_in,n_out,'tanh');

% hiddenlayer - softmax
n_in = 500;
n_out = 10;
[w1,b1] = h_o_layer_build(n_in,n_out,'softmax');

while(epoch<=n_epochs && looping)
    
    if(epoch == 0)
        fprintf('Training...\n');
    end
    
    for minibatch_index = [1 : 1 : n_train_batches] 
        iter = epoch * n_train_batches + minibatch_index - 1;
        
        if(rem(iter,100)==0)
            fprintf('training iter = %d\n',iter);
        end
        % make minibatch size data
        % x = 784*500
        x = train_x((minibatch_index-1)*batch_size+1:minibatch_index*batch_size,:).';
        y = train_y((minibatch_index-1)*batch_size+1:minibatch_index*batch_size);
        y = y + 1;
        
        % hiddenlayer - tanh
        % output1 -> layer2_input = 20000*500
        % output2 = 500*500
        layer0_input = x;
        [output0] = hiddenlayer(layer0_input,w0,b0,'tanh');
        
        % hiddenlayer - softmax
        % output2 -> layer3_input = 500*500
        % p_y_given_x = 10*500
        layer1_input = output0;
        [p_y_given_x,pred_y] = h_o_layer(layer1_input,w1,b1,'softmax');
        
        % negative_log_likelihood
        [cost] = negative_log_likelihood(p_y_given_x,y);
  
        % updates
        [w1,b1,delta1] = h_o_updates(learning_rate,w1,b1,layer1_input,p_y_given_x,y,'softmax');
        [w0,b0,delta0] = h_h_updates(learning_rate,w0,b0,layer0_input,output0,w1,delta1,'tanh');
        % validation check
        if(rem(iter+1,validation_frequency) == 0)
            validation_losses = zeros(n_valid_batches,1);
            
            for minibatch_valid = [1 : 1 : n_valid_batches]
                x = valid_x((minibatch_valid-1)*batch_size+1:minibatch_valid*batch_size,:).';
                y = valid_y((minibatch_valid-1)*batch_size+1:minibatch_valid*batch_size);
                y = y + 1;
                
                % hiddenlayer - tanh
                % output1 -> layer2_input = 20000*500
                % output2 = 500*500
                layer0_input = x;
                [output0] = hiddenlayer(layer0_input,w0,b0,'tanh');
                
                % hiddenlayer - softmax
                % output2 -> layer3_input = 500*500
                % p_y_given_x = 10*500
                layer1_input = output0;
                [p_y_given_x,pred_y] = h_o_layer(layer1_input,w1,b1,'softmax');
                
                [errors] = error_calc(pred_y.',y);
                
                validation_losses(minibatch_valid) = errors;
            end
            
            this_validation_loss = mean(validation_losses);
            
            fprintf('epoch %d, minibatch %d/%d, validation error %f\n',epoch+1,minibatch_index,n_train_batches,this_validation_loss*100);
            
            if(this_validation_loss < best_validation_loss*improvement_threshold)
                patience = max(patience, iter*patience_increase);
                
                best_validation_loss = this_validation_loss;
                best_iter = iter;
                
                % test check
                test_losses = zeros(n_test_batches,1);
                
                for minibatch_test = [1 : 1 : n_test_batches]
                    x = test_x((minibatch_test-1)*batch_size+1:minibatch_test*batch_size,:).';
                    y = test_y((minibatch_test-1)*batch_size+1:minibatch_test*batch_size);
                    y = y + 1;
                    
                    % hiddenlayer - tanh
                    % output1 -> layer2_input = 20000*500
                    % output2 = 500*500
                    layer0_input = x;
                    [output0] = hiddenlayer(layer0_input,w0,b0,'tanh');
                    
                    % hiddenlayer - softmax
                    % output2 -> layer3_input = 500*500
                    % p_y_given_x = 10*500
                    layer1_input = output0;
                    [p_y_given_x,pred_y] = h_o_layer(layer1_input,w1,b1,'softmax');
                    
                    [errors] = error_calc(pred_y.',y);
                    
                    test_losses(minibatch_test) = errors;
                end
                
                test_score = mean(test_losses);
                fprintf('epoch %d, minibatch %d/%d, test error of best model %f\n',epoch+1,minibatch_index,n_train_batches,test_score*100);
            end
        end
        
        if(patience <= iter)
            looping = 0;
        end
    end
    
    epoch = epoch+1;
end

fprintf('Optimization Complete.\n');
fprintf('Optimization Complete.\n');
fprintf('Optimization Complete.\n');
fprintf('Optimization Complete.\n');