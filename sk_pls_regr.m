function out  = sk_pls_regr(X,Y,ncomp,scaling,nfold,nrepeats,nperm)

    out.original_data_X = X;
    out.original_data_Y = Y;

    [n,px] = size(X);
    [n,py] = size(Y);
    N = 1:n;
    
    Xm = repmat(mean(X),n,1);
    Ym = repmat(mean(Y),n,1);
    
    Xc = X - Xm;
    Yc = Y - Ym;
    
    out.Xm = Xm;
    out.Xc = Xc;
 
    out.Ym = Ym;
    out.Yc = Yc;
    
    out.sq_sum = sum(sum(Y.^2))/(n*py);
    out.sq_sumc = sum(sum(Yc.^2))/(n*py);
    
    if scaling == true
        Xstd = repmat(std(Xc),n,1);
        Ystd = repmat(std(Yc),n,1);
        Xsc = Xc./Xstd;
        Ysc = Yc./Ystd;
    else
        Xstd = ones(n,px);
        Ystd = ones(n,py);
        Xsc = Xc;
        Ysc = Yc;
    end
    
    out.Xstd = Xstd;
    out.Xsc = Xsc;
    
    out.Ystd = Ystd;
    out.Ysc = Ysc;
        
    [Px,Py,Tx,Ty,beta] = plsregress(Xsc,Ysc,ncomp);
    
    out.Px_train = Px;
    out.Tx_train = Tx;
    out.Py_train = Py;
    out.Ty_train = Ty;
    out.beta_train = beta;
    
    out.pred_train = ([ones(n,1), out.Xsc]*out.beta_train).*out.Ystd + out.Ym;    
    out.sq_error_train = sum(sum((Y-out.pred_train).^2))/(n*py);
    out.var_sq_error_train = sum((Y - out.pred_train).^2)/n;
    out.mean_sq_error_train = mean(out.var_sq_error_train);
    out.R2 = 1 - out.sq_error_train/(sum(sum(Yc.^2))/(n*py));
    out.var_R2 = 1 - out.var_sq_error_train./(sum(Yc.^2)/n);
    out.mean_R2 = 1 - out.mean_sq_error_train./(sum(sum(Yc.^2))/(n*py));
    
    cvrepeats = cell(nrepeats,4);
    cv_sq_error_train = zeros(nrepeats,1);
    cv_var_sq_error_train = zeros(nrepeats,py);
    cv_mean_sq_error_train = zeros(nrepeats,1);
    cv_sq_error_test = zeros(nrepeats,1);
    cv_var_sq_error_test = zeros(nrepeats,py);
    cv_mean_sq_error_test = zeros(nrepeats,1);
    
    for i = 1:nrepeats
        
        cvsets = cell(nfold,5);
        temp_n = N;
        
        for j = 1:nfold

            if length(temp_n) >= ceil(n/nfold)
                cvsets{j,1} = datasample(temp_n,ceil(n/nfold),'Replace',false);
                cvsets{j,1} = sort(cvsets{j,1},'ascend');
            else
                cvsets{j,1} = temp_n;
            end
            temp_n(ismember(temp_n,cvsets{j,1})) = [];
            cvsets{j,2} = N(~ismember(N,cvsets{j,1}));
            cvsets{j,2} = sort(cvsets{j,2},'ascend');
            
            cv_pls = sk_pls_regr_cv(X,Y,ncomp,scaling,cvsets(j,:));
            cvsets{j,3} = cv_pls.pred_train;
            cvsets{j,4} = cv_pls.pred_test;
            
            cv_sq_error_train(i,1) = cv_sq_error_train(i,1) + cv_pls.sq_error_train;
            cv_var_sq_error_train(i,:) = cv_var_sq_error_train(i,:) + cv_pls.var_sq_error_train;
            cv_mean_sq_error_train(i,1) = cv_mean_sq_error_train(i,1) + mean(cv_pls.var_sq_error_train);
            
            cv_sq_error_test(i,1) = cv_sq_error_test(i,1) + cv_pls.sq_error_cv;
            cv_var_sq_error_test(i,:) = cv_var_sq_error_test(i,:) + cv_pls.var_sq_error_cv;
            
        end
        
        cvrepeats{i,1} = cvsets;
        
        cv_sq_error_train(i,1) = cv_sq_error_train(i,1)/nfold;
        cv_var_sq_error_train(i,:) = cv_var_sq_error_train(i,:)/nfold;
        cv_mean_sq_error_train(i,1) = cv_mean_sq_error_train(i,1)/nfold;
        
        cv_sq_error_test(i,1) = cv_sq_error_test(i,1)/(n*py);
        cv_var_sq_error_test(i,:) = cv_var_sq_error_test(i,:)/n;
        cv_mean_sq_error_test(i,1) = mean(cv_var_sq_error_test(i,:));
        
        cvrepeats{i,2} = cv_sq_error_test(i,1);
        cvrepeats{i,3} = cv_var_sq_error_test(i,:);
        cvrepeats{i,4} = cv_mean_sq_error_test(i,1);
        
    end
    
    out.cvsets = cvrepeats;
    
    out.cv_sq_error_train = mean(cv_sq_error_train);
    out.cv_var_sq_error_train = mean(cv_var_sq_error_train);
    out.cv_mean_sq_error_train = mean(cv_mean_sq_error_train);
    
    out.cv_sq_error_test = mean(cv_sq_error_test);
    out.cv_var_sq_error_test = mean(cv_var_sq_error_test);
    out.cv_mean_sq_error_test = mean(cv_mean_sq_error_test);
    
    out.Q2_train = 1 - out.cv_sq_error_train/sum(sum(Yc.^2))/(n*py);
    out.var_Q2_train = 1 - out.cv_var_sq_error_train./(sum(Yc.^2)/n);
    out.mean_Q2_train = 1 - out.cv_mean_sq_error_train./(sum(sum(Yc.^2))/(n*py));
    
    out.Q2 = 1 - out.cv_sq_error_test/(sum(sum(Yc.^2))/(n*py));
    out.var_Q2 = 1 - out.cv_var_sq_error_test./(sum(Yc.^2)/n);
    out.mean_Q2 = 1 - out.cv_mean_sq_error_test./(sum(sum(Yc.^2))/(n*py));

    if nperm > 0
    
        out.perm_cv_error = zeros(nperm,1);
        out.perm_Q2 = zeros(nperm,1);
        out.perm_var_Q2 = zeros(nperm,py);
        out.perm_mean_Q2 = zeros(nperm,1);
        
        for k = 1:nperm
           
            clc;
            disp(k);
            
            indx = datasample(1:n,n,'Replace',false);
            Xperm = X;
            Yperm = Y(indx,:);
  
            cv_sq_error_test = zeros(nrepeats,1);
            cv_var_sq_error_test = zeros(nrepeats,py);
            cv_mean_sq_error_test = zeros(nrepeats,1);
            
            try
            
            for i = 1:nrepeats
        
                cvsets = cell(nfold,2);
                temp_n = N;
                for j = 1:nfold

                    if length(temp_n) >= ceil(n/nfold)
                        cvsets{j,1} = datasample(temp_n,ceil(n/nfold),'Replace',false);
                        cvsets{j,1} = sort(cvsets{j,1},'ascend');
                    else
                        cvsets{j,1} = temp_n;
                    end
                    temp_n(ismember(temp_n,cvsets{j,1})) = [];
                    cvsets{j,2} = N(~ismember(N,cvsets{j,1}));
                    cvsets{j,2} = sort(cvsets{j,2},'ascend');
                    
                    
                    cv_pls = sk_pls_regr_cv(Xperm,Yperm,ncomp,scaling,cvsets(j,:));

                    cv_sq_error_test(i,1) = cv_sq_error_test(i,1) + cv_pls.sq_error_cv;
                    cv_var_sq_error_test(i,:) = cv_var_sq_error_test(i,:) + cv_pls.var_sq_error_cv;

                end

                cv_sq_error_test(i,1) = cv_sq_error_test(i,1)/(n*py);
                cv_var_sq_error_test(i,:) = cv_var_sq_error_test(i,:)/n;
                cv_mean_sq_error_test(i,1) = mean(cv_var_sq_error_test(i,:));
       
            end
            

            catch
                disp([num2str(k) ' : error. Go on?']);
                pause;
                k = k - 1;
            end

            out.perm_Q2(k,1) = 1 - mean(cv_sq_error_test)/(sum(sum(Yc.^2))/(n*py));
            out.perm_var_Q2(k,:) = 1 - mean(cv_var_sq_error_test)./(sum(Yc.^2)/n);
            out.perm_mean_Q2(k,1) = 1 - mean(cv_mean_sq_error_test)/(sum(sum(Yc.^2))/(n*py));
            
        end
        
    end
    
end