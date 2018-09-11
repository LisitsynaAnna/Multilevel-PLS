function out = sk_pls_regr_cv(X,Y,ncomp,scaling, cvset)

    %cvset{1,1}
    %cvset{1,2}

    out.original_data_X = X;
    out.original_data_Y = Y;
    
    [~,~] = size(X);
    [n,py] = size(Y);
    N = 1:n;
    
    indx_train = find(ismember(N, cvset{1,2}));
    indx_test  = find(ismember(N, cvset{1,1}));
    
    X_train = X(indx_train,:);
    X_test  = X(indx_test,:);
    Y_train = Y(indx_train,:);
    Y_test  = Y(indx_test,:);
    
    out.X_train = X_train;
    out.X_test = X_test;
    out.Y_train = Y_train;
    out.Y_test = Y_test;
    
    [n_train,px] = size(X_train);
    [n_test,px]  = size(X_test);
    [n_train,py] = size(Y_train);
    
    Xm_train = repmat(mean(X_train),n_train,1);
    Ym_train = repmat(mean(Y_train),n_train,1);
    
    Xc_train = X_train - Xm_train;
    Yc_train = Y_train - Ym_train;
    
    out.Xm_train = Xm_train;
    out.Xc_train = Xc_train;

    out.Ym_train = Ym_train;
    out.Yc_train = Yc_train;
    
    if scaling == true
        Xstd_train = repmat(std(Xc_train),n_train,1);
        Ystd_train = repmat(std(Yc_train),n_train,1);
        Xsc_train = Xc_train./Xstd_train;
        Ysc_train = Yc_train./Ystd_train;
    else
        Xstd_train = ones(n_train,px);
        Ystd_train = ones(n_train,py);
        Xsc_train = Xc_train;
        Ysc_train = Yc_train;
    end
    
    out.Xstd_train = Xstd_train;
    out.Xsc_train = Xsc_train;
    
    out.Ystd_train = Ystd_train;
    out.Ysc_train = Ysc_train;
    
    [Px_train, Py_train, Tx_train, Ty_train, beta_train] = plsregress(Xsc_train, Ysc_train, ncomp);
    
    out.Px_train = Px_train;
    out.Tx_train = Tx_train;
    out.Py_train = Py_train;
    out.Ty_train = Ty_train;
    out.beta_train = beta_train;
    
    out.pred_train = ([ones(n_train,1), out.Xsc_train]*out.beta_train).*out.Ystd_train + out.Ym_train;   
    out.sq_error_train = sum(sum((Y_train - out.pred_train).^2))/(n_train*py);
    out.var_sq_error_train = sum((Y_train - out.pred_train).^2)/n_train;
    
    Xc_test = X_test - out.Xm_train(1:n_test,:);
    
    Xsc_test = Xc_test./Xstd_train(1:n_test,:);
    
    out.Xc_test = Xc_test;
    out.Yc_train = Yc_train(1:n_test,:);
    
    out.pred_test = ([ones(n_test,1), Xsc_test]*out.beta_train).*out.Ystd_train(1:n_test,:) + out.Ym_train(1:n_test,:);
    out.sq_error_cv = sum(sum((Y_test - out.pred_test).^2));
    out.var_sq_error_cv = sum((Y_test - out.pred_test).^2,1);

end