function out  = sk_mm_pls_Xw(X,Y,subj,ncomp,scaling,nfold,nrepeats,nperm)

    out.original_data_X = X;
    out.original_data_Y = Y;
    out.original_data_subj = subj;

    [n,p] = size(X);
    usubj = unique(subj);
    s = length(usubj);
    
    % split-up variation ( X = Xm + Xb + Xc )
    
    Xm = repmat(mean(X),n,1);
    Ym = repmat(mean(Y),n,1);
    
    Xc = X - Xm;
    Yc = Y - Ym;
    
    Xb = zeros(n,p);
    for i = 1:s
        Xb(subj == usubj(i),:) = repmat(mean(Xc(subj == usubj(i),:)), sum(subj == usubj(i)), 1);
    end
    
    Xw = Xc - Xb;
    
    out.Xm = Xm;
    out.Xc = Xc;
    out.Xb = Xb;
    out.Xw = Xw;
    
    out.Ym = Ym;
    out.Yc = Yc;
    
    if scaling == true
        Xwstd = repmat(std(Xw),n,1);
        Ycstd = repmat(std(Yc),n,1);
        Xwsc = Xw./Xwstd;
        Ycsc = Yc./Ycstd;
    else
        Xwstd = 1;
        Ycstd = 1;
        Xwsc = Xw/Xwstd;
        Ycsc = Yc/Ycstd;
    end
    
    out.Xwstd = Xwstd;
    out.Xwsc = Xwsc;
    
    out.Ycstd = Ycstd;
    out.Ycsc = Ycsc;
        
    [Px,Py,Tx,Ty,beta] = plsregress(Xwsc,Ycsc,ncomp);
    
    out.Px_train = Px;
    out.Tx_train = Tx;
    out.Py_train = Py;
    out.Ty_train = Ty;
    out.beta_train = beta;
    out.pred_train = ([ones(n,1), out.Xwsc]*out.beta_train).*out.Ycstd + out.Ym;
    out.sq_error_train = sum((Y-out.pred_train).^2)/n;
    out.sq_error_mod_train = sum((Y-(2./(1+exp(-2*out.pred_train))-1)).^2)/n;
    out.R2 = 1 - out.sq_error_train/(sum(Y'*Y)/n);
    
    cvrepeats = cell(nrepeats,5);
    cv_sq_error_train = zeros(nrepeats,1);
    cv_sq_error_mod_train = zeros(nrepeats,1);
    cv_sq_error_test = zeros(nrepeats,1);
    cv_sq_error_mod_test = zeros(nrepeats,1);
    
    for i = 1:nrepeats
        
        cvsets = cell(nfold,6);
        temp_usubj = usubj;
        for j = 1:nfold

            if length(temp_usubj) >= ceil(s/nfold)
                cvsets{j,1} = datasample(temp_usubj,ceil(s/nfold),'Replace',false);
            else
                cvsets{j,1} = temp_usubj;
            end
            temp_usubj(ismember(temp_usubj,cvsets{j,1})) = [];
            cvsets{j,2} = usubj(~ismember(usubj,cvsets{j,1}));
            
            cv_pls = sk_mm_pls_cv_Xw(X,Y,subj,ncomp,scaling,cvsets(j,:));
            cvsets{j,4} = cv_pls.pred_train;
            cvsets{j,6} = cv_pls.pred_test;

            cv_sq_error_train(i,1) = cv_sq_error_train(i,1) + cv_pls.sq_error_train;
            cv_sq_error_mod_train(i,1) = cv_sq_error_mod_train(i,1) + cv_pls.sq_error_mod_train;
            cv_sq_error_test(i,1) = cv_sq_error_test(i,1) + cv_pls.sq_error_cv;
            cv_sq_error_mod_test(i,1) = cv_sq_error_mod_test(i,1) + cv_pls.sq_error_mod_cv;
            
        end
        cvrepeats{i,1} = cvsets;
        cv_sq_error_train(i,1) = cv_sq_error_train(i,1)/nfold;
        cv_sq_error_mod_train(i,1) = cv_sq_error_mod_train(i,1)/nfold;
        cv_sq_error_test(i,1) = cv_sq_error_test(i,1)/n;
        cv_sq_error_mod_test(i,1) = cv_sq_error_mod_test(i,1)/n;
        cvrepeats{i,3} = cv_sq_error_test(i,1);
        cvrepeats{i,4} = cv_sq_error_mod_test(i,1);
        
    end
    out.cvsets = cvrepeats;
    out.cv_sq_error_train = mean(cv_sq_error_train);
    out.cv_sq_error_mod_train = mean(cv_sq_error_mod_train);
    out.cv_sq_error_test = mean(cv_sq_error_test);
    out.cv_sq_error_mod_test = mean(cv_sq_error_mod_test);
    out.Q2 = 1 - out.cv_sq_error_test/(sum(Y'*Y)/n);

    if nperm > 0
    
        out.perm_Q2 = zeros(nperm,1);
        
        for k = 1:nperm
           
            clc;
            disp(k);
            
            indx = datasample(1:n,n,'Replace',false);
            Xperm = X(indx,:);
            Yperm = Y;
  
            cv_sq_error_test = zeros(nrepeats,1);
            
            for i = 1:nrepeats
        
                cvsets = cell(nfold,6);
                temp_usubj = usubj;
                for j = 1:nfold

                    if length(temp_usubj) >= ceil(s/nfold)
                        cvsets{j,1} = datasample(temp_usubj,ceil(s/nfold),'Replace',false);
                    else
                        cvsets{j,1} = temp_usubj;
                    end
                    temp_usubj(ismember(temp_usubj,cvsets{j,1})) = [];
                    cvsets{j,2} = usubj(~ismember(usubj,cvsets{j,1}));
            
                    cv_pls = sk_mm_pls_cv_Xw(Xperm,Yperm,subj,ncomp,scaling,cvsets(j,:));
                    cvsets{j,4} = cv_pls.pred_train;
                    cvsets{j,6} = cv_pls.pred_test;

                    cv_sq_error_test(i,1) = cv_sq_error_test(i,1) + cv_pls.sq_error_cv;
            
                end

                cv_sq_error_test(i,1) = cv_sq_error_test(i,1)/n;
        
            end

            out.perm_Q2(k,1) = 1 - mean(cv_sq_error_test)/(sum(Y'*Y)/n);
            
        end
        
    end
    
end