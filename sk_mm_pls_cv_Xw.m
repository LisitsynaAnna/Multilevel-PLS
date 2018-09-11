function out = sk_mm_pls_cv_Xw(X,Y,subj,ncomp,scaling, cvset)

    %cvset{1,1}
    %cvset{1,2}

    out.original_data_X = X;
    out.original_data_Y = Y;
    out.original_data_subj = subj;
    
    [n,p] = size(X);
    usubj = unique(subj);
    s = length(usubj);
    
    indx_train = find(ismember(subj, cvset{1,2}));
    indx_test  = find(ismember(subj, cvset{1,1}));
    
    X_train = X(indx_train,:);
    X_test  = X(indx_test,:);
    Y_train = Y(indx_train,:);
    Y_test  = Y(indx_test,:);
    
    subj_train = subj(indx_train,:);
    subj_test  = subj(indx_test,:);
    usubj_train = unique(subj_train);
    usubj_test = unique(subj_test);
    s_train = length(usubj_train);
    s_test = length(usubj_test);
    
    out.X_train = X_train;
    out.X_test = X_test;
    out.Y_train = Y_train;
    out.Y_test = Y_test;
    out.subj_train = subj_train;
    out.subj_test = subj_test;
    
    [n_train,~] = size(X_train);
    [n_test,~]  = size(X_test);
    
    Xm_train = repmat(mean(X_train),n_train,1);
    Ym_train = repmat(mean(Y_train),n_train,1);
    
    Xc_train = X_train - Xm_train;
    Yc_train = Y_train - Ym_train;
    
    Xb_train = zeros(n_train,p);
    for i = 1:s_train
        Xb_train(subj_train == usubj_train(i),:) = repmat(mean(Xc_train(subj_train == usubj_train(i),:)), sum(subj_train == usubj_train(i)), 1);
    end
    
    Xw_train = Xc_train - Xb_train;
    
    out.Xm_train = Xm_train;
    out.Xc_train = Xc_train;
    out.Xb_train = Xb_train;
    out.Xw_train = Xw_train;
    
    out.Ym_train = Ym_train;
    out.Yc_train = Yc_train;
    out.subj_train = subj_train;
    
    if scaling == true
        Xwstd_train = repmat(std(Xw_train),n_train,1);
        Ycstd_train = repmat(std(Yc_train),n_train,1);
        Xwsc_train = Xw_train./Xwstd_train;
        Ycsc_train = Yc_train./Ycstd_train;
    else
        Xwstd_train = 1;
        Ycstd_train = 1;
        Xwsc_train = Xw_train/Xwstd_train;
        Ycsc_train = Yc_train/Ycstd_train;
    end
    
    out.Xwstd_train = Xwstd_train;
    out.Xwsc_train = Xwsc_train;
    
    out.Ycstd_train = Ycstd_train;
    out.Ycsc_train = Ycsc_train;
    
    [Px_train, Py_train, Tx_train, Ty_train, beta_train] = plsregress(Xwsc_train, Ycsc_train, ncomp);
    
    out.Px_train = Px_train;
    out.Tx_train = Tx_train;
    out.Py_train = Py_train;
    out.Ty_train = Ty_train;
    out.beta_train = beta_train;
    out.pred_train = ([ones(n_train,1), out.Xwsc_train]*out.beta_train).*out.Ycstd_train + out.Ym_train;    
    out.sq_error_train = sum((Y_train - out.pred_train).^2)/n_train;
    out.sq_error_mod_train = sum((Y_train - (2./(1+exp(-2*out.pred_train))-1)).^2)/n_train;
    
    
    
    Xc_test = X_test - out.Xm_train(1:n_test,:);
    
    Xb_test = zeros(n_test,p);
    for i = 1:s_test
        Xb_test(subj_test == usubj_test(i),:) = repmat(mean(Xc_test(subj_test == usubj_test(i),:)), sum(subj_test == usubj_test(i)), 1);
    end
    
    Xw_test = Xc_test - Xb_test;
    
    Xwsc_test = Xw_test./Xwstd_train(1:n_test,:);
    
    out.Xc_test = Xc_test;
    out.Xb_test = Xb_test;
    out.subj_test = subj_test;
    out.pred_test = ([ones(n_test,1), Xwsc_test]*out.beta_train).*out.Ycstd_train(1:n_test,1) + out.Ym_train(1:n_test,1);
    out.sq_error_cv = sum((Y_test - out.pred_test).^2);
    out.sq_error_mod_cv = sum((Y_test - (2./(1+exp(-2*out.pred_test))-1)).^2);

end