clear all;

% read data
[train_x,train_y,valid_x,valid_y,test_x,test_y] = loadpickle();

learning_rate = 0.1;
n_epochs = 200;
nkerns = [20 50];
batch_size = 10;

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

% LogisticRegression
n_in = 28*28;
n_out = 10;
[w,b] = LR_build(n_in,n_out);

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
        x = train_x((minibatch_index-1)*batch_size+1:minibatch_index*batch_size,:);
        y = train_y((minibatch_index-1)*batch_size+1:minibatch_index*batch_size);
        y = y + 1;
        
        [p_y_given_x,pred_y] = LogisticRegression(x,w,b);
        
        % negative_log_likelihood
        [cost] = negative_log_likelihood(p_y_given_x,y);
        
        % updates
        [w,b,del] = h_o_updates(batch_size,learning_rate,w,b,x,p_y_given_x,y);
        
        % validation check
        if(rem(iter+1,validation_frequency) == 0)
            validation_losses = zeros(n_valid_batches,1);
            
            for i = [1 : 1 : n_valid_batches]
                x = valid_x((i-1)*batch_size+1:i*batch_size,:);
                y = valid_y((i-1)*batch_size+1:i*batch_size);
                y = y + 1;
                
                [p_y_given_x,pred_y] = LogisticRegression(x,w,b);
                
                [errors] = error_calc(pred_y,y);
                
                validation_losses(i) = errors;
            end
            
            this_validation_loss = mean(validation_losses);
            
            fprintf('epoch %d, minibatch %d/%d, validation error %f\n',epoch+1,minibatch_index,n_train_batches,this_validation_loss*100);
            
            if(this_validation_loss < best_validation_loss*improvement_threshold)
                patience = max(patience, iter*patience_increase);
                
                best_validation_loss = this_validation_loss;
                best_iter = iter;
                
                % test check
                test_losses = zeros(n_test_batches,1);
                
                for i = [1 : 1 : n_test_batches]
                    x = test_x((i-1)*batch_size+1:i*batch_size,:);
                    y = test_y((i-1)*batch_size+1:i*batch_size);
                    y = y + 1;
                    
                    [p_y_given_x,pred_y] = LogisticRegression(x,w,b);
                    
                    [errors] = error_calc(pred_y,y);
                    
                    test_losses(i) = errors;
                end
                
                test_score = mean(test_losses);
                fprintf('epoch %d, minibatch %d/%d, test error of best model %f\n',epoch+1,minibatch_index,n_train_batches,test_score*100);
              
                learning_rate = learning_rate / 2;
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
%{
###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
%}