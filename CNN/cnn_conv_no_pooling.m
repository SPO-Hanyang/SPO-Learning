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
validation_frequency = 10;

best_validation_loss = 999999;
best_iter = 0;
test_score = 0;

looping = 1;
epoch = 0;

%Build Layers
fprintf('Building...\n');

% 1st convlayer - tanh
image_shape0 = [28 28 1 batch_size];
filter_shape0 = [5 5 1 nkerns(1)];
[w0, b0] = convlayer_build(filter_shape0);

% 2nd convlayer - tanh
image_shape1 = [24 24 20 500];
filter_shape1 = [5 5 nkerns(1) nkerns(2)];
[w1, b1] = convlayer_build(filter_shape1);

% hiddenlayer - tanh
n_in = nkerns(2)*20*20;
n_out = 500;
[w2,b2] = hiddenlayer_build(n_in,n_out,'tanh');

% hiddenlayer - softmax
n_in = 500;
n_out = 10;
[w3,b3] = h_o_layer_build(n_in,n_out,'softmax');

while(epoch<=n_epochs && looping)
    
    if(epoch == 0)
        fprintf('Training...\n');
    end
    
    for minibatch_index = [1 : 1 : n_train_batches] 
        iter = epoch * n_train_batches + minibatch_index - 1;
        
        if(rem(iter,10)==0)
            fprintf('training iter = %d\n',iter);
        end
        % make minibatch size data
        % x = 784*500
        x = train_x((minibatch_index-1)*batch_size+1:minibatch_index*batch_size,:).';
        y = train_y((minibatch_index-1)*batch_size+1:minibatch_index*batch_size);
        y = y + 1;
        
        % 1st convlayer - tanh
        % x -> lyaer0_input = 28*28*500
        % output0 = 24*24*20*500
        layer0_input = reshape(x,image_shape0);
        [output0] = convlayer(layer0_input,image_shape0,filter_shape0,w0,b0);
        
        % 2nd convlayer - tanh
        % output0 -> layer1_input = 24*24*20*500
        % output1 = 20*20*50*500
        layer1_input = reshape(output0,image_shape1);
        [output1] = convlayer(layer1_input,image_shape1,filter_shape1,w1,b1);
        
        % hiddenlayer - tanh
        % output1 -> layer2_input = 20000*500
        % output2 = 500*500
        out_size = size(output1);
        layer2_input = reshape(output1,[prod(out_size(1:3)) out_size(4)]);
        [output2] = hiddenlayer(layer2_input,w2,b2,'tanh');
        
        % hiddenlayer - softmax
        % output2 -> layer3_input = 500*500
        % p_y_given_x = 10*500
        layer3_input = output2;
        [p_y_given_x,pred_y] = h_o_layer(layer3_input,w3,b3,'softmax');
        
        % negative_log_likelihood
        [cost] = negative_log_likelihood(p_y_given_x,y);
  
        % updates
        tic
        [w3,b3,delta3] = h_o_updates(learning_rate,w3,b3,layer3_input,p_y_given_x,y,'softmax');
        toc
        tic
        [w2,b2,delta2] = h_h_updates(learning_rate,w2,b2,layer2_input,output2,w3,delta3,'tanh');
        toc
        tic
        [w1,b1,delta1] = c_h_updates(learning_rate,filter_shape1,w1,b1,layer1_input,output1,w2,delta2,'tanh');
        toc
        tic
        [w0,b0,delta0] = c_c_updates(learning_rate,filter_shape0,w0,b0,layer0_input,output0,filter_shape1,w1,delta1,'tanh');
        toc
        % validation check
        if(rem(iter+1,validation_frequency) == 0)
            validation_losses = zeros(n_valid_batches,1);
            
            for minibatch_valid = [1 : 1 : n_valid_batches]
                x = valid_x((minibatch_valid-1)*batch_size+1:minibatch_valid*batch_size,:).';
                y = valid_y((minibatch_valid-1)*batch_size+1:minibatch_valid*batch_size);
                y = y + 1;
                
                % 1st convlayer - tanh
                layer0_input = reshape(x,image_shape0);
                [output0] = convlayer(layer0_input,image_shape0,filter_shape0,w0,b0);
                
                % 2nd convlayer - tanh
                layer1_input = reshape(output0,image_shape1);
                [output1] = convlayer(layer1_input,image_shape1,filter_shape1,w1,b1);
                
                % hiddenlayer - tanh
                out_size = size(output1);
                layer2_input = reshape(output1,[prod(out_size(1:3)) out_size(4)]);
                [output2] = hiddenlayer(layer2_input,w2,b2,'tanh');
                
                % hiddenlayer - softmax
                layer3_input = output2;
                [p_y_given_x,pred_y] = h_o_layer(layer3_input,w3,b3,'softmax');
                
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
                    
                    % 1st convlayer - tanh
                    layer0_input = reshape(x,image_shape0);
                    [output0] = convlayer(layer0_input,image_shape0,filter_shape0,w0,b0);
                    
                    % 2nd convlayer - tanh
                    layer1_input = reshape(output0,image_shape1);
                    [output1] = convlayer(layer1_input,image_shape1,filter_shape1,w1,b1);
                    
                    % hiddenlayer - tanh
                    out_size = size(output1);
                    layer2_input = reshape(output1,[prod(out_size(1:3)) out_size(4)]);
                    [output2] = hiddenlayer(layer2_input,w2,b2,'tanh');
                    
                    % hiddenlayer - softmax
                    layer3_input = output2;
                    [p_y_given_x,pred_y] = h_o_layer(layer3_input,w3,b3,'softmax');
                    
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